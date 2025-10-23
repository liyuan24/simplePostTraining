import torch
from typing import Optional


def masked_mean(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: int = None,
) -> torch.Tensor:
    """
    Compute the mean of the tensor along the specified dimensions, ignoring the dimensions where the mask is 0.
    Args:
        tensor: The tensor to compute the mean of.
        mask: The mask to apply to the tensor.
        dim: The dimensions to compute the mean along.
    Returns:
        The mean of the tensor along the specified dimensions, ignoring the dimensions where the mask is 0.
    """
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim)
