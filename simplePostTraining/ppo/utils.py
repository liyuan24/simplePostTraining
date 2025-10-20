import torch


def compute_kl(
    log_probs: torch.tensor, ref_log_probs: torch.tensor
):
    """
    See: http://joschu.net/blog/kl-approx.html
    Compute the KL divergence between the log probabilities and the reference log probabilities.
    Args:
        log_probs: The log probabilities. Shape: (batch_size, seq_len)
        ref_log_probs: The reference log probabilities. Shape: (batch_size, seq_len)
    Returns:
        The KL divergence. Shape: (batch_size, seq_len)
    """
    # the k1 approximation of KL, Monte Carlo estimation
    return log_probs.float() - ref_log_probs.float()


def compute_reward(
    reward_at_end: torch.tensor,
    kl_coeff: float,
    kl: torch.tensor,
    action_mask: torch.tensor,
):
    """
    Compute the reward for the given KL divergence and action mask.
    Args:
        reward_at_end: The reward at the end of the episode. Shape: (batch_size,)
        kl_coeff: The coefficient for the KL divergence.
        kl: The per-token KL divergence. Shape: (batch_size, seq_len)
        action_mask: The action mask, only the LLM response tokens are 1. Shape: (batch_size, seq_len)
    Returns:
        The reward. Shape: (batch_size, seq_len)
    """
    if kl_coeff < 0:
        kl_coeff = 0
    res = -kl_coeff * kl
    # add reward only for the last token (EOS token position)
    # action_mask is 0 for prompt and padded tokens, 1 for generated tokens
    # Find the last position where action_mask is 1 for each sequence
    last_token_positions = (
        action_mask.size(1)
        - 1
        - torch.argmax(action_mask.flip(dims=[1]), dim=1)
    )
    batch_indices = torch.arange(
        action_mask.size(0), device=action_mask.device
    )
    res[
        batch_indices, last_token_positions
    ] += reward_at_end
    return res
