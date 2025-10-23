import torch
from typing import List


class Dataloader:
    """
    Data loader for PPO training with prompt sampling capabilities.
    """

    def __init__(
        self,
    ):
        """
        Initialize the dataloader.
        """

    def sample_prompts(self) -> List[torch.Tensor]:
        """
        Sample prompt tokens for PPO training.

        Returns:
            List of prompt tensors
        """
        # For now, return simple hardcoded prompts
        # In practice, this could sample from a dataset, use templates, etc.
        prompt_tokens = [
            torch.tensor([0, 1]),  # First prompt
            torch.tensor([0, 2]),  # Second prompt
        ]

        return prompt_tokens
