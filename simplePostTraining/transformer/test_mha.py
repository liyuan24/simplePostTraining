import torch
from simplePostTraining.transformer.mha import (
    MultiHeadSelfAttention,
)


def test_self_attention_output_shape():
    """Test that MultiHeadSelfAttention produces correct output shape."""
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8

    # Create self-attention module
    self_attn = MultiHeadSelfAttention(
        d_model=d_model, num_heads=num_heads, dropout=0.1
    )

    # Create input tensor
    x = torch.randn(batch_size, seq_len, d_model)

    # Forward pass
    output = self_attn(x)

    # Check output shape
    assert output.shape == (
        batch_size,
        seq_len,
        d_model,
    ), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
