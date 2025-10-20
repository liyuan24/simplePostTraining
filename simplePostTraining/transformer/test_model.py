import torch
from simplePostTraining.transformer.model import MLP, Block


def test_mlp_output_shape():
    """Test that MLP produces correct output shape."""
    batch_size = 2
    seq_len = 10
    d_model = 512
    d_ff = 2048

    # Create MLP module
    mlp = MLP(d_model=d_model, d_ff=d_ff, dropout=0.1)

    # Create input tensor
    x = torch.randn(batch_size, seq_len, d_model)

    # Forward pass
    output = mlp(x)

    # Check output shape
    assert output.shape == (
        batch_size,
        seq_len,
        d_model,
    ), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"


def test_block_output_shape():
    """Test that Block produces correct output shape."""
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8

    # Create Block module
    block = Block(
        d_model=d_model,
        num_heads=num_heads,
        dropout=0.1,
        norm_first=True,
    )

    # Create input tensor
    x = torch.randn(batch_size, seq_len, d_model)

    # Forward pass
    output = block(x)

    # Check output shape
    assert output.shape == (
        batch_size,
        seq_len,
        d_model,
    ), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"


def test_block_with_swiglu():
    """Test that Block with SwiGLU activation works correctly."""
    batch_size = 2
    seq_len = 10
    d_model = 256
    num_heads = 4

    # Create Block module with SwiGLU
    block = Block(
        d_model=d_model,
        num_heads=num_heads,
        dropout=0.1,
        activation="swiglu",
        norm_first=True,
    )

    # Create input tensor
    x = torch.randn(batch_size, seq_len, d_model)

    # Forward pass
    output = block(x)

    # Check output shape
    assert output.shape == (
        batch_size,
        seq_len,
        d_model,
    ), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
