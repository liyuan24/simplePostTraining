import torch
import torch.nn.functional as F
from typing import Optional


def sample_tokens(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """
    Sample tokens with optional top-k and top-p filtering.
    Both top_k and top_p can be used simultaneously.

    Args:
        logits: Logits tensor of shape (batch_size, vocab_size)
        temperature: Sampling temperature (higher = more random)
        top_k: Number of top tokens to consider (None = no filtering)
        top_p: Cumulative probability threshold (None = no filtering)

    Returns:
        Sampled token indices of shape (batch_size,)
    """
    # Apply temperature scaling
    scaled_logits = logits / temperature

    # Apply top-k filtering first (if specified)
    if top_k is not None:
        top_k = min(top_k, logits.size(-1))
        topk_logits, topk_indices = torch.topk(
            scaled_logits, top_k, dim=-1
        )

        # Create mask for top-k tokens
        mask = torch.full_like(scaled_logits, float("-inf"))
        mask.scatter_(-1, topk_indices, topk_logits)
        scaled_logits = mask

    # Apply top-p filtering (if specified)
    if top_p is not None:
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(
            scaled_logits, descending=True, dim=-1
        )

        # Calculate cumulative probabilities
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        # Find cutoff point where cumulative probability exceeds p
        sorted_indices_to_remove = cumulative_probs > top_p

        # Shift mask to keep at least one token
        sorted_indices_to_remove[..., 1:] = (
            sorted_indices_to_remove[..., :-1].clone()
        )
        sorted_indices_to_remove[..., 0] = 0

        # Create mask for tokens to keep
        indices_to_remove = (
            sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
        )

        # Apply mask
        scaled_logits[indices_to_remove] = float("-inf")

    # Sample from final distribution
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(
        -1
    )


# Example usage and testing
if __name__ == "__main__":
    # Test sampling functions
    batch_size = 2
    vocab_size = 10
    logits = torch.randn(batch_size, vocab_size)

    print("Original logits:")
    print(logits)
    print()

    # Test different sampling methods
    print("Temperature sampling (temp=1.0):")
    tokens1 = sample_tokens(logits, temperature=1.0)
    print(tokens1)

    print("Temperature sampling (temp=0.5):")
    tokens2 = sample_tokens(logits, temperature=0.5)
    print(tokens2)

    print("Top-k sampling (k=3):")
    tokens3 = sample_tokens(
        logits, temperature=1.0, top_k=3
    )
    print(tokens3)

    print("Top-p sampling (p=0.9):")
    tokens4 = sample_tokens(
        logits, temperature=1.0, top_p=0.9
    )
    print(tokens4)

    print("Combined sampling (top_k=3):")
    tokens5 = sample_tokens(
        logits, temperature=1.0, top_k=3
    )
    print(tokens5)

    print("Combined sampling (top_k=3, top_p=0.9):")
    tokens6 = sample_tokens(
        logits, temperature=1.0, top_k=3, top_p=0.9
    )
    print(tokens6)
