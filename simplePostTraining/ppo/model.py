import torch
from typing import Optional
from simplePostTraining.transformer import Transformer
from simplePostTraining.utils.sampling import sample_tokens


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
        prompt_tokens: torch.Tensor,
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

        Args:
            prompt_tokens: Starting prompt tokens of shape (batch_size, prompt_len)
            max_response_length: Maximum response length
            temperature: Sampling temperature
            top_k: Number of top tokens to consider (None = no filtering)
            top_p: Cumulative probability threshold (None = no filtering)

        Returns:
            Tuple of (sequences, log_probs, values, action_mask)
        """
        device = next(self.policy.parameters()).device
        batch_size, prompt_len = prompt_tokens.shape

        # Start with prompt tokens
        sequences = prompt_tokens.clone()

        # Track which sequences are still generating (not EOS)
        active_mask = torch.ones(
            batch_size, dtype=torch.bool, device=device
        )

        # Store log probabilities for each step
        log_probs_list = []

        # Generate sequences step by step
        remaining_length = max_response_length
        for step in range(remaining_length):
            # Only generate for active sequences
            if not active_mask.any():
                break

            # Get logits only for active sequences
            active_indices = active_mask.nonzero(
                as_tuple=True
            )[0]
            active_sequences = sequences[active_indices]
            active_logits = self.get_logits(
                active_sequences
            )

            # Sample next tokens for active sequences only
            active_next_tokens = sample_tokens(
                active_logits, temperature, top_k, top_p
            )

            # Calculate log probabilities for active sequences
            active_step_log_probs = torch.log_softmax(
                active_logits, dim=-1
            )
            active_step_log_probs = (
                active_step_log_probs.gather(
                    1, active_next_tokens.unsqueeze(-1)
                ).squeeze(-1)
            )

            # Create full-size tensors for all sequences
            next_tokens = torch.zeros(
                batch_size, device=device, dtype=torch.long
            )
            step_log_probs = torch.zeros(
                batch_size, device=device
            )

            # Fill in values for active sequences
            next_tokens[active_indices] = active_next_tokens
            step_log_probs[active_indices] = (
                active_step_log_probs
            )

            log_probs_list.append(step_log_probs)

            # Check for EOS tokens before appending
            eos_mask = next_tokens == self.eos_token

            # For sequences that hit EOS, use EOS token; for others, use sampled tokens
            next_tokens = torch.where(
                eos_mask,
                torch.full_like(
                    next_tokens, self.eos_token
                ),
                next_tokens,
            )

            # Append next tokens to sequences
            sequences = torch.cat(
                [sequences, next_tokens.unsqueeze(-1)],
                dim=1,
            )

            # Update active mask (sequences that hit EOS become inactive)
            active_mask = active_mask & (~eos_mask)

        # Stack log probabilities: (batch_size, num_actions)
        if log_probs_list:
            log_probs = torch.stack(
                log_probs_list, dim=1
            )  # (batch_size, num_actions)
        else:
            log_probs = torch.zeros(
                batch_size, 0, device=device
            )

        # Pad sequences to max_response_length if needed
        current_seq_len = sequences.shape[1]
        max_total_length = prompt_len + max_response_length
        if current_seq_len < max_total_length:
            # Pad with EOS tokens (or a special padding token)
            padding_length = (
                max_total_length - current_seq_len
            )
            padding_tokens = torch.full(
                (batch_size, padding_length),
                self.eos_token,
                device=device,
                dtype=sequences.dtype,
            )
            sequences = torch.cat(
                [sequences, padding_tokens], dim=1
            )

        # Create action mask: 1 for actions, 0 for prompts and padding
        action_mask = torch.zeros_like(
            sequences, dtype=torch.float
        )

        # Mark generated tokens as actions, but stop at EOS
        for i in range(batch_size):
            # Find the first EOS token in the generated part
            generated_part = sequences[i, prompt_len:]
            eos_positions = (
                generated_part == self.eos_token
            ).nonzero(as_tuple=True)[0]

            if len(eos_positions) > 0:
                # Stop at the first EOS token
                first_eos_pos = eos_positions[0].item()
                action_mask[
                    i,
                    prompt_len : prompt_len
                    + first_eos_pos
                    + 1,
                ] = 1.0
            else:
                # No EOS found, mark all generated tokens as actions
                action_mask[i, prompt_len:] = 1.0

        # Calculate values
        values = self._calculate_values(sequences)

        return sequences, log_probs, values, action_mask

    def get_values(
        self, sequences: torch.Tensor
    ) -> torch.Tensor:
        """
        Get value estimates from the value model.

        Args:
            sequences: Generated sequences of shape (batch_size, seq_len)

        Returns:
            Value estimates of shape (batch_size,)
        """
        # Get hidden states from the value model
        hidden_states = self.value_model.get_hidden_states(
            sequences
        )  # (batch_size, seq_len, d_model)

        # Use the last token's hidden state for value prediction
        last_hidden = hidden_states[
            :, -1, :
        ]  # (batch_size, d_model)

        # Project to single scalar value
        values = self.value_head(last_hidden).squeeze(
            -1
        )  # (batch_size,)
        return values

    def _calculate_values(
        self, sequences: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate values for sequences using the value model.

        Args:
            sequences: Generated sequences of shape (batch_size, seq_len)

        Returns:
            Values of shape (batch_size,)
        """
        return self.get_values(sequences)

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
            Log probabilities of shape (batch_size, num_actions)
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

        # Extract only action log probs using the mask
        action_mask_shifted = action_mask[
            :, 1:
        ]  # Shift mask to align with next tokens
        action_log_probs = (
            all_log_probs * action_mask_shifted
        )

        # Remove non-action positions (where mask is 0)
        action_log_probs_list = []
        for i in range(batch_size):
            action_positions = action_mask_shifted[i] == 1
            if action_positions.any():
                action_log_probs_list.append(
                    action_log_probs[i][action_positions]
                )
            else:
                action_log_probs_list.append(
                    torch.tensor(
                        [], device=sequences.device
                    )
                )

        # Pad to same length
        max_actions = max(
            len(probs) for probs in action_log_probs_list
        )
        if max_actions > 0:
            padded_log_probs = torch.zeros(
                batch_size,
                max_actions,
                device=sequences.device,
            )
            for i, probs in enumerate(
                action_log_probs_list
            ):
                if len(probs) > 0:
                    padded_log_probs[i, : len(probs)] = (
                        probs
                    )
            return padded_log_probs
        else:
            return torch.zeros(
                batch_size, 0, device=sequences.device
            )

    def train_step(
        self,
        sequences: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        action_mask: torch.Tensor,
        clip_ratio: float = 0.2,
    ) -> dict:
        """
        Perform one PPO training step.

        Args:
            sequences: Generated sequences
            old_log_probs: Old log probabilities
            advantages: Advantage estimates
            returns: Return estimates
            clip_ratio: PPO clipping ratio

        Returns:
            Dictionary with training metrics
        """
        # Policy optimization
        self.policy_optimizer.zero_grad()

        # Get current log probabilities
        current_log_probs = self.get_action_log_probs(
            sequences, action_mask
        )

        # Calculate ratio
        ratio = torch.exp(current_log_probs - old_log_probs)

        # Expand advantages to match action dimensions
        num_actions = current_log_probs.shape[1]
        advantages_expanded = advantages.unsqueeze(
            1
        ).expand(-1, num_actions)

        # PPO clipped objective
        surr1 = ratio * advantages_expanded
        surr2 = (
            torch.clamp(
                ratio, 1 - clip_ratio, 1 + clip_ratio
            )
            * advantages_expanded
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
        values_fresh = self.get_values(sequences_fresh)
        value_loss = torch.nn.functional.mse_loss(
            values_fresh, returns.detach()
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

    # Create prompt tokens (batch of starting sequences)
    prompt_tokens = torch.tensor(
        [[0, 1], [0, 2]]
    )  # (batch_size=2, prompt_len=2)

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

    # Training step
    advantages = values - values.mean()
    returns = values
    metrics = model.train_step(
        sequences,
        log_probs,
        advantages,
        returns,
        action_mask,
    )
    print("Training metrics:", metrics)
