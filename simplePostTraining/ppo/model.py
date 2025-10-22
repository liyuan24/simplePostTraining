import torch
from typing import Optional
from simplePostTraining.transformer import Transformer
from simplePostTraining.utils.sampling import sample_tokens
from simplePostTraining.ppo.utils import (
    compute_kl,
    compute_reward,
)


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
    ) -> tuple[
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

        Returns:
            Tuple of (sequences, log_probs, values, action_mask)
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

        return (
            padded_sequences,
            padded_log_probs,
            padded_values,
            padded_action_masks,
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

    def train_step(
        self,
        sequences: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_values: torch.Tensor,
        action_mask: torch.Tensor,
        ref_log_probs: torch.Tensor,
        reward_at_end: torch.Tensor,
        kl_coeff: float = 0.1,
        clip_ratio: float = 0.2,
    ) -> dict:
        """
        Perform one PPO training step.

        Args:
            sequences: Generated sequences
            old_log_probs: Old log probabilities
            old_values: Old value estimates (per token)
            action_mask: Action mask
            ref_log_probs: Reference log probabilities for KL penalty
            reward_at_end: Scalar reward at end of each sequence
            kl_coeff: KL divergence coefficient
            clip_ratio: PPO clipping ratio

        Returns:
            Dictionary with training metrics
        """
        # Policy optimization
        self.policy_optimizer.zero_grad()

        # Get current log probabilities and values
        current_log_probs = self.get_action_log_probs(
            sequences, action_mask
        )
        current_values = self._calculate_values_per_token(
            sequences
        )

        # Calculate KL divergence and rewards using utility functions
        kl = compute_kl(current_log_probs, ref_log_probs)
        rewards = compute_reward(
            reward_at_end, kl_coeff, kl, action_mask
        )

        # Calculate advantages
        advantages = rewards - old_values

        # Calculate ratio
        ratio = torch.exp(current_log_probs - old_log_probs)

        # PPO clipped objective
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(
                ratio, 1 - clip_ratio, 1 + clip_ratio
            )
            * advantages
        )
        actor_loss = -torch.min(surr1, surr2).mean()

        # Policy backward pass
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), max_norm=1.0
        )
        self.policy_optimizer.step()

        # Value optimization
        self.value_optimizer.zero_grad()

        # Value loss (MSE) - use fresh sequences to avoid double backward
        sequences_fresh = sequences.detach().clone()
        values_fresh = self._calculate_values_per_token(
            sequences_fresh
        )
        value_loss = torch.nn.functional.mse_loss(
            values_fresh, rewards.detach()
        )

        # Value backward pass
        value_loss.backward()
        value_params = list(
            self.value_model.parameters()
        ) + list(self.value_head.parameters())
        torch.nn.utils.clip_grad_norm_(
            value_params, max_norm=1.0
        )
        self.value_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "value_loss": value_loss.item(),
            "ratio_mean": ratio.mean().item(),
            "ratio_std": ratio.std().item(),
        }


# Example usage
if __name__ == "__main__":
    # Create model
    model = PPOModel()

    # Create prompt tokens (list of starting sequences)
    # sample some prompts
    prompt_tokens = [
        torch.tensor([0, 1]),  # First prompt
        torch.tensor([0, 2]),  # Second prompt
    ]

    # Generate rollout starting from prompts
    sequences, log_probs, values, action_mask = (
        model.generate_rollout(
            prompt_tokens=prompt_tokens,
            max_response_length=4,  # Maximum response length (excluding prompt)
            temperature=0.8,
            top_k=5,
        )
    )
    print("Prompt tokens:", prompt_tokens)
    print("Generated sequences:", sequences)
    print("Log probabilities:", log_probs)
    print("Values:", values)
    print("Action mask:", action_mask)

    # Create reference model (typically a frozen copy of the policy model)
    ref_model = Transformer(
        vocab_size=10,
        d_model=8,
        num_heads=2,
        num_layers=2,
        max_seq_len=10,
    )
    ref_model.eval()  # Set to evaluation mode

    # Generate reference log probabilities using the reference model
    # The reference model should evaluate the same sequences that the policy generated
    with torch.no_grad():
        # Get logits for all positions except the last (same as policy model)
        ref_logits = ref_model(
            sequences[:, :-1]
        )  # (batch_size, seq_len-1, vocab_size)

        # Get log probabilities for the actual next tokens (same as policy model)
        next_tokens = sequences[
            :, 1:
        ]  # (batch_size, seq_len-1)
        ref_log_probs = torch.log_softmax(
            ref_logits, dim=-1
        )
        ref_all_log_probs = ref_log_probs.gather(
            2, next_tokens.unsqueeze(-1)
        ).squeeze(
            -1
        )  # (batch_size, seq_len-1)

        # Create full-length log probabilities tensor (same as policy model)
        batch_size, seq_len = sequences.shape
        ref_full_log_probs = torch.zeros(
            batch_size, seq_len, device=sequences.device
        )
        ref_log_probs[:, 1:] = ref_all_log_probs

    print("Reference log probabilities:", ref_log_probs)

    # Create end-of-sequence rewards (scalar per sequence)
    reward_at_end = torch.ones(
        sequences.size(0)
    )  # Reward of 1.0 for each sequence

    # Training step
    metrics = model.train_step(
        sequences,
        log_probs,
        values,
        action_mask,
        ref_log_probs,
        reward_at_end,
        kl_coeff=0.1,
    )
    print("Training metrics:", metrics)
