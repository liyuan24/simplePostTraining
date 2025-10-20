import torch
import pytest
from simplePostTraining.ppo.utils import (
    compute_reward,
    compute_kl,
)


# Tests for compute_kl
def test_compute_kl_basic():
    """Test basic KL divergence computation."""
    batch_size = 2
    seq_len = 5

    log_probs = torch.tensor(
        [
            [-0.5, -0.6, -0.7, -0.8, -0.9],
            [-1.0, -1.1, -1.2, -1.3, -1.4],
        ]
    )
    ref_log_probs = torch.tensor(
        [
            [-0.4, -0.5, -0.6, -0.7, -0.8],
            [-0.9, -1.0, -1.1, -1.2, -1.3],
        ]
    )

    result = compute_kl(log_probs, ref_log_probs)

    # Check shape
    assert result.shape == (batch_size, seq_len)

    # Check correctness: KL should be log_probs - ref_log_probs
    expected = log_probs - ref_log_probs
    assert torch.allclose(result, expected)


def test_compute_kl_zero_divergence():
    """Test when log_probs equals ref_log_probs (zero KL divergence)."""
    batch_size = 3
    seq_len = 4

    log_probs = torch.tensor(
        [
            [-0.5, -0.6, -0.7, -0.8],
            [-1.0, -1.1, -1.2, -1.3],
            [-0.3, -0.4, -0.5, -0.6],
        ]
    )
    ref_log_probs = log_probs.clone()

    result = compute_kl(log_probs, ref_log_probs)

    # When log probs are identical, KL should be zero
    assert torch.allclose(
        result, torch.zeros(batch_size, seq_len)
    )


def test_compute_kl_positive_divergence():
    """Test when log_probs > ref_log_probs (positive KL)."""
    batch_size = 2
    seq_len = 3

    # log_probs are higher (less negative) than ref_log_probs
    log_probs = torch.tensor(
        [
            [-0.1, -0.2, -0.3],
            [-0.4, -0.5, -0.6],
        ]
    )
    ref_log_probs = torch.tensor(
        [
            [-0.5, -0.6, -0.7],
            [-0.8, -0.9, -1.0],
        ]
    )

    result = compute_kl(log_probs, ref_log_probs)

    # KL should be positive when log_probs > ref_log_probs
    assert torch.all(result > 0)

    # Verify exact values
    expected = log_probs - ref_log_probs
    assert torch.allclose(result, expected)


def test_compute_kl_negative_divergence():
    """Test when log_probs < ref_log_probs (negative KL)."""
    batch_size = 2
    seq_len = 3

    # log_probs are lower (more negative) than ref_log_probs
    log_probs = torch.tensor(
        [
            [-1.0, -1.1, -1.2],
            [-2.0, -2.1, -2.2],
        ]
    )
    ref_log_probs = torch.tensor(
        [
            [-0.5, -0.6, -0.7],
            [-1.0, -1.1, -1.2],
        ]
    )

    result = compute_kl(log_probs, ref_log_probs)

    # KL should be negative when log_probs < ref_log_probs
    assert torch.all(result < 0)

    # Verify exact values
    expected = log_probs - ref_log_probs
    assert torch.allclose(result, expected)


def test_compute_kl_different_dtypes():
    """Test that compute_kl converts to float regardless of input dtype."""
    batch_size = 2
    seq_len = 3

    # Create tensors with different dtypes
    log_probs = torch.tensor(
        [
            [-1, -2, -3],
            [-4, -5, -6],
        ],
        dtype=torch.int32,
    )
    ref_log_probs = torch.tensor(
        [
            [-0.5, -1.5, -2.5],
            [-3.5, -4.5, -5.5],
        ],
        dtype=torch.float64,
    )

    result = compute_kl(log_probs, ref_log_probs)

    # Check that result is float
    assert result.dtype == torch.float32

    # Verify correctness
    expected = log_probs.float() - ref_log_probs.float()
    assert torch.allclose(result, expected)


def test_compute_kl_single_sequence():
    """Test with batch size 1."""
    log_probs = torch.tensor([[-0.5, -0.6, -0.7]])
    ref_log_probs = torch.tensor([[-0.4, -0.5, -0.6]])

    result = compute_kl(log_probs, ref_log_probs)

    assert result.shape == (1, 3)
    expected = log_probs - ref_log_probs
    assert torch.allclose(result, expected)


def test_compute_kl_large_values():
    """Test with large absolute values."""
    batch_size = 2
    seq_len = 4

    log_probs = torch.tensor(
        [
            [-100.0, -200.0, -300.0, -400.0],
            [-500.0, -600.0, -700.0, -800.0],
        ]
    )
    ref_log_probs = torch.tensor(
        [
            [-50.0, -150.0, -250.0, -350.0],
            [-450.0, -550.0, -650.0, -750.0],
        ]
    )

    result = compute_kl(log_probs, ref_log_probs)

    # Check shape
    assert result.shape == (batch_size, seq_len)

    # Verify correctness
    expected = log_probs - ref_log_probs
    assert torch.allclose(result, expected)


def test_compute_kl_mixed_signs():
    """Test with mixed positive and negative values."""
    batch_size = 2
    seq_len = 4

    log_probs = torch.tensor(
        [
            [0.5, -0.5, 1.0, -1.0],
            [2.0, -2.0, 0.0, -0.1],
        ]
    )
    ref_log_probs = torch.tensor(
        [
            [0.3, -0.7, 0.8, -1.2],
            [1.5, -2.5, -0.2, 0.1],
        ]
    )

    result = compute_kl(log_probs, ref_log_probs)

    # Verify correctness
    expected = log_probs - ref_log_probs
    assert torch.allclose(result, expected)


# Tests for compute_reward
def test_compute_reward_basic():
    """Test basic reward computation with simple case."""
    batch_size = 2
    seq_len = 5

    # Create simple inputs
    reward_at_end = torch.tensor([1.0, 2.0])
    kl_coeff = 0.1
    kl = torch.ones(batch_size, seq_len) * 0.5
    action_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1],  # Last token at position 4
            [0, 1, 1, 1, 0],  # Last token at position 3
        ]
    )

    result = compute_reward(
        reward_at_end, kl_coeff, kl, action_mask
    )

    # Check shape
    assert result.shape == (batch_size, seq_len)

    # Check that KL penalty is applied everywhere
    expected_kl_penalty = -kl_coeff * 0.5
    assert torch.allclose(
        result[:, 0],
        torch.tensor(
            [expected_kl_penalty, expected_kl_penalty]
        ),
    )

    # Check that reward is added only at the last token position
    # For batch 0, last token is at position 4
    assert torch.isclose(
        result[0, 4],
        torch.tensor(expected_kl_penalty + 1.0),
    )
    # For batch 1, last token is at position 3
    assert torch.isclose(
        result[1, 3],
        torch.tensor(expected_kl_penalty + 2.0),
    )

    # Check that other positions only have KL penalty
    assert torch.isclose(
        result[0, 0], torch.tensor(expected_kl_penalty)
    )
    assert torch.isclose(
        result[1, 0], torch.tensor(expected_kl_penalty)
    )


def test_compute_reward_last_position():
    """Test that reward is correctly added to the last position where action_mask is 1."""
    batch_size = 3
    seq_len = 6

    reward_at_end = torch.tensor([10.0, 20.0, 30.0])
    kl_coeff = 0.2
    kl = torch.zeros(batch_size, seq_len)
    action_mask = torch.tensor(
        [
            [0, 0, 0, 1, 1, 1],  # Last token at position 5
            [0, 0, 1, 1, 0, 0],  # Last token at position 3
            [0, 1, 1, 1, 1, 1],  # Last token at position 5
        ]
    )

    result = compute_reward(
        reward_at_end, kl_coeff, kl, action_mask
    )

    # Since kl is zero, result should be zero everywhere except at last token positions
    assert torch.isclose(result[0, 5], torch.tensor(10.0))
    assert torch.isclose(result[1, 3], torch.tensor(20.0))
    assert torch.isclose(result[2, 5], torch.tensor(30.0))

    # Check other positions are zero
    assert torch.isclose(result[0, 0], torch.tensor(0.0))
    assert torch.isclose(result[1, 0], torch.tensor(0.0))
    assert torch.isclose(result[2, 0], torch.tensor(0.0))


def test_compute_reward_negative_kl_coeff():
    """Test that negative kl_coeff is clamped to 0."""
    batch_size = 2
    seq_len = 4

    reward_at_end = torch.tensor([5.0, 10.0])
    kl_coeff = -0.5  # Negative coefficient
    kl = torch.ones(batch_size, seq_len) * 2.0
    action_mask = torch.tensor(
        [
            [0, 1, 1, 1],  # Last token at position 3
            [0, 0, 1, 1],  # Last token at position 3
        ]
    )

    result = compute_reward(
        reward_at_end, kl_coeff, kl, action_mask
    )

    # Since kl_coeff is negative and should be clamped to 0, kl penalty should be 0
    # So only reward should be added at last positions
    assert torch.isclose(result[0, 3], torch.tensor(5.0))
    assert torch.isclose(result[1, 3], torch.tensor(10.0))
    assert torch.isclose(result[0, 0], torch.tensor(0.0))


def test_compute_reward_varying_kl():
    """Test with varying KL divergence values."""
    batch_size = 2
    seq_len = 4

    reward_at_end = torch.tensor([1.0, 2.0])
    kl_coeff = 0.5
    kl = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
        ]
    )
    action_mask = torch.tensor(
        [
            [0, 1, 1, 1],  # Last token at position 3
            [0, 1, 1, 0],  # Last token at position 2
        ]
    )

    result = compute_reward(
        reward_at_end, kl_coeff, kl, action_mask
    )

    # Check KL penalty is correctly applied
    assert torch.isclose(
        result[0, 0], torch.tensor(-0.5 * 0.1)
    )
    assert torch.isclose(
        result[0, 1], torch.tensor(-0.5 * 0.2)
    )
    assert torch.isclose(
        result[1, 0], torch.tensor(-0.5 * 0.5)
    )

    # Check reward is added at last token position
    assert torch.isclose(
        result[0, 3], torch.tensor(-0.5 * 0.4 + 1.0)
    )
    assert torch.isclose(
        result[1, 2], torch.tensor(-0.5 * 0.7 + 2.0)
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
