import torch
from typing import Optional
from simplePostTraining.transformer import Transformer
from simplePostTraining.utils.sampling import sample_tokens
from simplePostTraining.ppo.utils import (
    compute_kl,
    compute_reward,
)
from simplePostTraining.ppo.dataloader import Dataloader
from simplePostTraining.utils.utils import masked_mean


class PPOModel:
    """
    PPO model wrapper for transformer-based language model.

    Args:
        vocab_size: Vocabulary size (default: 10)
        d_model: Model dimension (default: 8)
        num_heads: Number of attention heads (default: 2)
        num_layers: Number of transformer layers (default: 2)
        max_seq_len: Maximum sequence length (default: 10)
        eos_token: End-of-sequence token ID (default: 9)
    """

    def __init__(
        self,
        vocab_size: int = 10,
        d_model: int = 8,
        num_heads: int = 2,
        num_layers: int = 2,
        max_seq_len: int = 10,
        eos_token: int = 9,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.eos_token = eos_token

        # Initialize policy model (transformer)
        self.policy = Transformer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            dropout=0.1,
            activation="relu",
            is_causal=True,
            norm_first=True,
        )

        # Initialize value model (transformer + value head)
        self.value_model = Transformer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            dropout=0.1,
            activation="relu",
            is_causal=True,
            norm_first=True,
        )

        # Value head: projects to single scalar value
        self.value_head = torch.nn.Linear(d_model, 1)

        # Initialize reference model for KL penalty
        self.ref_model = Transformer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            dropout=0.1,
            activation="relu",
            is_causal=True,
            norm_first=True,
        )
        self.ref_model.eval()  # Set to evaluation mode

        # Initialize optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=1e-4,
            weight_decay=0.01,
        )

        # Value optimizer includes both transformer and value head
        value_params = list(
            self.value_model.parameters()
        ) + list(self.value_head.parameters())
        self.value_optimizer = torch.optim.Adam(
            value_params,
            lr=1e-4,
            weight_decay=0.01,
        )

    def forward(
        self, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the policy model.

        Args:
            input_ids: Token indices of shape (batch_size, seq_len)

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        return self.policy(input_ids)

    def get_logits(
        self, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Get logits for the last token in each sequence.

        Args:
            input_ids: Token indices of shape (batch_size, seq_len)

        Returns:
            Logits for last tokens of shape (batch_size, vocab_size)
        """
        logits = self.forward(input_ids)
        return logits[:, -1, :]  # Last token logits

    def sample_action(
        self,
        input_ids: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Sample next token from the model.

        Args:
            input_ids: Current sequence of shape (batch_size, seq_len)
            temperature: Sampling temperature
            top_k: Number of top tokens to consider (None = no filtering)
            top_p: Cumulative probability threshold (None = no filtering)

        Returns:
            Next token indices of shape (batch_size,)
        """
        logits = self.get_logits(input_ids)
        return sample_tokens(
            logits, temperature, top_k, top_p
        )

    def generate_rollout(
        self,
        prompt_tokens: list[torch.Tensor],
        max_response_length: int = 4,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        gamma: float = 0.99,
        lambda_: float = 0.95,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Generate rollout sequences for PPO training starting from prompt tokens.
        Processes sequences one by one for better memory efficiency.

        Args:
            prompt_tokens: List of prompt tensors, each of shape (prompt_len,)
            max_response_length: Maximum response length
            temperature: Sampling temperature
            top_k: Number of top tokens to consider (None = no filtering)
            top_p: Cumulative probability threshold (None = no filtering)
            gamma: Discount factor for GAE
            lambda_: GAE parameter for bias-variance tradeoff

        Returns:
            Tuple of (sequences, log_probs, values, action_mask, advantages, returns)
        """
        device = next(self.policy.parameters()).device
        batch_size = len(prompt_tokens)

        # Store results for each sequence
        all_sequences = []
        all_log_probs = []
        all_values = []
        all_action_masks = []

        # Process each sequence individually
        for i in range(batch_size):
            prompt = prompt_tokens[i].unsqueeze(
                0
            )  # Add batch dimension

            # Generate single sequence
            sequence, log_probs, value, action_mask = (
                self._generate_single_rollout(
                    prompt,
                    max_response_length,
                    temperature,
                    top_k,
                    top_p,
                )
            )

            all_sequences.append(
                sequence.squeeze(0)
            )  # Remove batch dimension
            all_log_probs.append(
                log_probs.squeeze(0)
            )  # Remove batch dimension
            all_values.append(
                value.squeeze(0)
            )  # Remove batch dimension
            all_action_masks.append(
                action_mask.squeeze(0)
            )  # Remove batch dimension

        # Find maximum sequence length for padding
        max_seq_len = max(
            seq.shape[0] for seq in all_sequences
        )

        # Pad sequences to same length
        padded_sequences = torch.zeros(
            batch_size,
            max_seq_len,
            device=device,
            dtype=torch.long,
        )
        padded_action_masks = torch.zeros(
            batch_size,
            max_seq_len,
            device=device,
            dtype=torch.int,
        )
        padded_log_probs = torch.zeros(
            batch_size, max_seq_len, device=device
        )
        padded_values = torch.zeros(
            batch_size, max_seq_len, device=device
        )

        for i in range(batch_size):
            seq_len = all_sequences[i].shape[0]

            padded_sequences[i, :seq_len] = all_sequences[i]
            padded_action_masks[i, :seq_len] = (
                all_action_masks[i]
            )
            padded_log_probs[i, :seq_len] = all_log_probs[i]
            padded_values[i, :seq_len] = all_values[i]

        # Calculate rewards using compute_reward method
        # Calculate reference log probabilities using pre-initialized reference model
        with torch.no_grad():
            # Get reference logits for all positions except the last
            ref_logits = self.ref_model(
                padded_sequences[:, :-1]
            )

            # Get log probabilities for the actual next tokens
            next_tokens = padded_sequences[:, 1:]
            ref_log_probs = torch.log_softmax(
                ref_logits, dim=-1
            )
            ref_all_log_probs = ref_log_probs.gather(
                2, next_tokens.unsqueeze(-1)
            ).squeeze(-1)

            # Create full-length log probabilities tensor
            ref_log_probs = torch.zeros_like(
                padded_sequences, dtype=torch.float
            )
            ref_log_probs[:, 1:] = ref_all_log_probs

        # Calculate end-of-sequence rewards
        reward_at_end = self.get_reward_at_end(
            padded_sequences, padded_action_masks
        )

        # Calculate KL divergence
        kl = compute_kl(padded_log_probs, ref_log_probs)

        # Calculate rewards using utility function
        rewards = compute_reward(
            reward_at_end,
            0.1,  # kl_coeff = 0.1
            kl,
            padded_action_masks,
        )

        # Calculate GAE advantages and returns
        advantages, returns = self.calculate_gae(
            rewards,
            padded_values,
            padded_action_masks,
            gamma,
            lambda_,
        )

        return (
            padded_sequences,
            padded_log_probs,
            padded_values,
            padded_action_masks,
            advantages,
            returns,
        )

    def _generate_single_rollout(
        self,
        prompt: torch.Tensor,  # Shape: (1, prompt_len)
        max_response_length: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Generate rollout for a single sequence.

        Args:
            prompt: Single prompt of shape (1, prompt_len)
            max_response_length: Maximum response length (excluding prompt)
            temperature: Sampling temperature
            top_k: Number of top tokens to consider
            top_p: Cumulative probability threshold

        Returns:
            Tuple of (sequence, log_probs, value, action_mask)
        """
        device = prompt.device
        prompt_len = prompt.shape[1]

        # Start with prompt
        sequence = prompt.clone()
        log_probs_list = []

        # Generate response tokens one by one
        for step in range(max_response_length):
            # Get logits for current sequence
            logits = self.get_logits(sequence)

            # Sample next token
            next_token = sample_tokens(
                logits, temperature, top_k, top_p
            )

            # Calculate log probability
            step_log_probs = torch.log_softmax(
                logits, dim=-1
            )
            step_log_prob = step_log_probs.gather(
                1, next_token.unsqueeze(-1)
            ).squeeze(-1)

            log_probs_list.append(step_log_prob)

            # Append next token to sequence
            sequence = torch.cat(
                [sequence, next_token.unsqueeze(-1)], dim=1
            )

            # Check for EOS
            if next_token.item() == self.eos_token:
                break

        # Create log probabilities that match sequence length
        seq_len = sequence.shape[1]
        log_probs = torch.zeros(1, seq_len, device=device)

        # Fill in log probabilities for response tokens (after prompt)
        if log_probs_list:
            response_log_probs = torch.stack(
                log_probs_list, dim=1
            )  # (1, num_response_tokens)
            log_probs[
                0,
                prompt_len : prompt_len
                + len(log_probs_list),
            ] = response_log_probs[0]

        # Create action mask
        action_mask = torch.zeros_like(
            sequence, dtype=torch.int
        )
        action_mask[0, prompt_len:] = (
            1  # Mark all generated tokens as actions
        )

        # Calculate values for each token position
        values = self._calculate_values_per_token(sequence)

        return sequence, log_probs, values, action_mask

    def _calculate_values_per_token(
        self, sequences: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate values for each token position using the value model.

        Args:
            sequences: Generated sequences of shape (1, seq_len)

        Returns:
            Values of shape (1, seq_len)
        """
        # Get hidden states from the value model
        hidden_states = self.value_model.get_hidden_states(
            sequences
        )  # (1, seq_len, d_model)

        # Project to scalar values for each position
        values = self.value_head(hidden_states).squeeze(
            -1
        )  # (1, seq_len)

        return values

    def get_action_log_probs(
        self,
        sequences: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get log probabilities for actions in sequences.

        Args:
            sequences: Token sequences of shape (batch_size, seq_len)
            action_mask: Mask indicating action tokens of shape (batch_size, seq_len)

        Returns:
            Log probabilities of shape (batch_size, seq_len) - zeros for non-action positions
        """
        batch_size, seq_len = sequences.shape

        # Get logits for all positions except the last
        logits = self.forward(
            sequences[:, :-1]
        )  # (batch_size, seq_len-1, vocab_size)

        # Get log probabilities for the actual next tokens
        next_tokens = sequences[
            :, 1:
        ]  # (batch_size, seq_len-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        all_log_probs = log_probs.gather(
            2, next_tokens.unsqueeze(-1)
        ).squeeze(
            -1
        )  # (batch_size, seq_len-1)

        # Create full-length log probabilities tensor
        full_log_probs = torch.zeros(
            batch_size, seq_len, device=sequences.device
        )

        # Fill in log probabilities for next tokens (shifted by 1)
        full_log_probs[:, 1:] = all_log_probs

        # Apply action mask to zero out non-action positions
        action_log_probs = (
            full_log_probs * action_mask.float()
        )

        return action_log_probs

    def calculate_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float = 0.99,
        lambda_: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate Generalized Advantage Estimation (GAE) and returns.

        Args:
            rewards: Per-token rewards of shape (batch_size, seq_len)
            values: Value estimates of shape (batch_size, seq_len)
            action_mask: Action mask indicating action positions of shape (batch_size, seq_len)
            gamma: Discount factor for future rewards
            lambda_: GAE parameter for bias-variance tradeoff

        Returns:
            advantages: GAE advantages of shape (batch_size, seq_len)
            returns: Discounted returns of shape (batch_size, seq_len)
        """
        batch_size, seq_len = rewards.shape
        device = rewards.device

        # Initialize advantages and returns
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        # Calculate GAE for each sequence
        for i in range(batch_size):
            # Get rewards, values, and action mask for this sequence
            seq_rewards = rewards[i]  # (seq_len,)
            seq_values = values[i]  # (seq_len,)
            seq_action_mask = action_mask[i]  # (seq_len,)

            # Find the last action position using action_mask
            # This is where we should bootstrap the value
            last_action_idx = (
                seq_action_mask == 1
            ).nonzero(as_tuple=True)[0]
            if len(last_action_idx) > 0:
                last_idx = last_action_idx[-1].item()
            else:
                last_idx = seq_len - 1

            # Calculate advantages using GAE
            gae = 0
            for t in reversed(range(last_idx + 1)):
                if t == last_idx:
                    # At the last timestep, use the reward as the advantage
                    delta = seq_rewards[t] - seq_values[t]
                else:
                    # For other timesteps, use the standard GAE formula
                    delta = (
                        seq_rewards[t]
                        + gamma * seq_values[t + 1]
                        - seq_values[t]
                    )

                gae = delta + gamma * lambda_ * gae
                advantages[i, t] = gae
                returns[i, t] = (
                    seq_rewards[t] + seq_values[t]
                )

        return advantages, returns

    def get_reward_at_end(
        self,
        sequences: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate end-of-sequence rewards for each sequence.

        Args:
            sequences: Generated sequences of shape (batch_size, seq_len)
            action_mask: Action mask of shape (batch_size, seq_len)

        Returns:
            reward_at_end: Scalar reward for each sequence of shape (batch_size,)
        """
        batch_size = sequences.size(0)
        device = sequences.device

        # For now, return 1.0 for every sequence
        # In practice, this would be calculated based on sequence quality,
        # task completion, or other reward criteria
        reward_at_end = torch.ones(
            batch_size, device=device
        )

        return reward_at_end

    def train_step(
        self,
        sequences: torch.Tensor,
        old_log_probs: torch.Tensor,
        action_mask: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        num_epochs: int = 1,
        clip_ratio: float = 0.2,
        mini_batch_size: int = 1,
    ) -> dict:
        """
        Perform one PPO training step with shuffling and mini-batch processing.

        Args:
            sequences: Generated sequences
            old_log_probs: Old log probabilities
            action_mask: Action mask
            advantages: GAE advantages from rollout generation
            returns: GAE returns from rollout generation
            num_epochs: Number of training epochs
            clip_ratio: PPO clipping ratio
            mini_batch_size: Size of mini-batches for training

        Returns:
            Dictionary with training metrics
        """
        # Initialize metrics storage
        total_actor_loss = 0.0
        total_value_loss = 0.0
        total_ratio_mean = 0.0
        total_ratio_std = 0.0

        batch_size = sequences.size(0)
        device = sequences.device

        # Training loop for multiple epochs
        for epoch in range(num_epochs):
            # Shuffle the indices at the start of each epoch
            indices = torch.randperm(
                batch_size, device=device
            )
            shuffled_sequences = sequences[indices]
            shuffled_old_log_probs = old_log_probs[indices]
            shuffled_action_mask = action_mask[indices]
            shuffled_advantages = advantages[indices]
            shuffled_returns = returns[indices]

            # Process in mini-batches
            num_mini_batches = (
                batch_size + mini_batch_size - 1
            ) // mini_batch_size
            epoch_actor_loss = 0.0
            epoch_value_loss = 0.0
            epoch_ratio_mean = 0.0
            epoch_ratio_std = 0.0

            for mini_batch_idx in range(num_mini_batches):
                start_idx = mini_batch_idx * mini_batch_size
                end_idx = min(
                    start_idx + mini_batch_size, batch_size
                )

                # Get mini-batch data
                mini_sequences = shuffled_sequences[
                    start_idx:end_idx
                ]
                mini_old_log_probs = shuffled_old_log_probs[
                    start_idx:end_idx
                ]
                mini_action_mask = shuffled_action_mask[
                    start_idx:end_idx
                ]
                mini_advantages = shuffled_advantages[
                    start_idx:end_idx
                ]
                mini_returns = shuffled_returns[
                    start_idx:end_idx
                ]

                # Policy optimization
                self.policy_optimizer.zero_grad()

                # Get current log probabilities and values
                current_log_probs = (
                    self.get_action_log_probs(
                        mini_sequences, mini_action_mask
                    )
                )

                # Calculate ratio
                ratio = torch.exp(
                    current_log_probs
                    - mini_old_log_probs.detach()
                )

                # PPO clipped objective
                surr1 = ratio * mini_advantages.detach()
                surr2 = (
                    torch.clamp(
                        ratio,
                        1 - clip_ratio,
                        1 + clip_ratio,
                    )
                    * mini_advantages.detach()
                )
                actor_loss = masked_mean(
                    -torch.min(surr1, surr2),
                    mini_action_mask,
                    dim=1,
                ).mean()

                # Policy backward pass
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), max_norm=1.0
                )
                self.policy_optimizer.step()

                # Value optimization
                self.value_optimizer.zero_grad()

                # Value loss (MSE) - use fresh sequences to avoid double backward
                sequences_fresh = (
                    mini_sequences.detach().clone()
                )
                values_fresh = (
                    self._calculate_values_per_token(
                        sequences_fresh
                    )
                )
                value_loss = masked_mean(
                    (values_fresh - mini_returns.detach())
                    ** 2,
                    mini_action_mask,
                    dim=1,
                ).mean()

                # Value backward pass
                value_loss.backward()
                value_params = list(
                    self.value_model.parameters()
                ) + list(self.value_head.parameters())
                torch.nn.utils.clip_grad_norm_(
                    value_params, max_norm=1.0
                )
                self.value_optimizer.step()

                # Accumulate mini-batch metrics
                mini_batch_size_actual = (
                    mini_sequences.size(0)
                )
                epoch_actor_loss += (
                    actor_loss.item()
                    * mini_batch_size_actual
                )
                epoch_value_loss += (
                    value_loss.item()
                    * mini_batch_size_actual
                )
                epoch_ratio_mean += (
                    ratio.mean().item()
                    * mini_batch_size_actual
                )
                epoch_ratio_std += (
                    ratio.std().item()
                    * mini_batch_size_actual
                )

            # Average metrics for this epoch
            epoch_actor_loss /= batch_size
            epoch_value_loss /= batch_size
            epoch_ratio_mean /= batch_size
            epoch_ratio_std /= batch_size

            # Accumulate epoch metrics
            total_actor_loss += epoch_actor_loss
            total_value_loss += epoch_value_loss
            total_ratio_mean += epoch_ratio_mean
            total_ratio_std += epoch_ratio_std

        # Return average metrics across epochs
        return {
            "actor_loss": total_actor_loss / num_epochs,
            "value_loss": total_value_loss / num_epochs,
            "ratio_mean": total_ratio_mean / num_epochs,
            "ratio_std": total_ratio_std / num_epochs,
        }


# Example usage
if __name__ == "__main__":
    # Create model and dataloader
    model = PPOModel()
    dataloader = Dataloader()

    prompt_tokens = dataloader.sample_prompts()

    # Generate rollout starting from prompts
    (
        sequences,
        log_probs,
        values,
        action_mask,
        advantages,
        returns,
    ) = model.generate_rollout(
        prompt_tokens=prompt_tokens,
        max_response_length=4,  # Maximum response length (excluding prompt)
        temperature=0.8,
        top_k=5,
        gamma=0.99,
        lambda_=0.95,
    )
    print("Prompt tokens:", prompt_tokens)
    print("Generated sequences:", sequences)
    print("Log probabilities:", log_probs)
    print("Values:", values)
    print("Action mask:", action_mask)

    # Training step with mini-batch processing
    metrics = model.train_step(
        sequences,
        log_probs,
        action_mask,
        advantages,
        returns,
        num_epochs=2,
        mini_batch_size=1,  # Process one sequence at a time
    )
    print("Training metrics:", metrics)
