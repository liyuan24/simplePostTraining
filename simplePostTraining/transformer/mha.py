import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module as described in "Attention is All You Need".

    Args:
        d_model: The dimension of the model (input/output dimension)
        num_heads: Number of attention heads
        dropout: Dropout probability (default: 0.1)
        bias: Whether to use bias in linear projections (default: True)
        is_causal: If True, applies causal masking (default: False)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        is_causal: bool = False,
    ):
        super().__init__()
        assert (
            d_model % num_heads == 0
        ), "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = (
            d_model // num_heads
        )  # dimension per head
        self.dropout_p = dropout
        self.is_causal = is_causal

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(
            d_model, d_model, bias=bias
        )

        # Causal mask buffer
        self.register_buffer(
            "causal_mask", None, persistent=False
        )

    def _get_causal_mask(
        self,
        seq_len_q: int,
        seq_len_k: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate or retrieve cached causal mask.

        Args:
            seq_len_q: Query sequence length
            seq_len_k: Key sequence length
            device: Device to create mask on

        Returns:
            Causal mask of shape (seq_len_q, seq_len_k)
        """
        # Use cached mask if available and correct size
        if (
            self.causal_mask is not None
            and self.causal_mask.shape[0] >= seq_len_q
            and self.causal_mask.shape[1] >= seq_len_k
        ):
            return self.causal_mask[:seq_len_q, :seq_len_k]

        # Create new causal mask: upper triangle is -inf, lower triangle (including diagonal) is 0
        mask = torch.triu(
            torch.full(
                (seq_len_q, seq_len_k),
                float("-inf"),
                device=device,
            ),
            diagonal=1,
        )
        self.causal_mask = mask
        return mask

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of multi-head attention.

        Args:
            query: Query tensor of shape (batch_size, seq_len_q, d_model)
            key: Key tensor of shape (batch_size, seq_len_k, d_model)
            value: Value tensor of shape (batch_size, seq_len_v, d_model)
            attn_mask: Attention mask of shape (seq_len_q, seq_len_k) or
                      (batch_size, seq_len_q, seq_len_k). Values should be
                      0 for positions to attend to and -inf for masked positions.
            key_padding_mask: Padding mask of shape (batch_size, seq_len_k).
                            True values indicate positions to mask.

        Returns:
            output: Output tensor of shape (batch_size, seq_len_q, d_model)
        """
        # Assert that attn_mask is None if is_causal is True
        if self.is_causal:
            assert (
                attn_mask is None
            ), "attn_mask must be None when is_causal=True"

        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]

        # Linear projections
        Q = self.q_proj(
            query
        )  # (batch_size, seq_len_q, d_model)
        K = self.k_proj(
            key
        )  # (batch_size, seq_len_k, d_model)
        V = self.v_proj(
            value
        )  # (batch_size, seq_len_k, d_model)

        # Reshape and transpose for multi-head attention
        # (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_head)
        Q = Q.view(
            batch_size,
            seq_len_q,
            self.num_heads,
            self.d_head,
        ).transpose(1, 2)
        K = K.view(
            batch_size,
            seq_len_k,
            self.num_heads,
            self.d_head,
        ).transpose(1, 2)
        V = V.view(
            batch_size,
            seq_len_k,
            self.num_heads,
            self.d_head,
        ).transpose(1, 2)

        # Prepare attention mask for scaled_dot_product_attention
        # Generate causal mask if needed
        combined_mask = attn_mask
        if self.is_causal:
            combined_mask = self._get_causal_mask(
                seq_len_q, seq_len_k, query.device
            )

        # Combine with key_padding_mask if needed
        if key_padding_mask is not None:
            # key_padding_mask: (batch_size, seq_len_k)
            # Reshape to (batch_size, 1, 1, seq_len_k) for broadcasting
            padding_mask = key_padding_mask.unsqueeze(
                1
            ).unsqueeze(2)
            padding_mask = padding_mask.expand(
                batch_size, 1, seq_len_q, seq_len_k
            )
            # Convert boolean mask to additive mask
            padding_mask = torch.where(
                padding_mask,
                float("-inf"),
                0.0,
            )
            if combined_mask is None:
                combined_mask = padding_mask
            else:
                combined_mask = combined_mask + padding_mask

        # Use flash attention
        attn_output = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=combined_mask,
            dropout_p=(
                self.dropout_p if self.training else 0.0
            ),
        )

        # Reshape and concatenate heads
        # (batch_size, num_heads, seq_len_q, d_k) -> (batch_size, seq_len_q, d_model)
        attn_output = attn_output.transpose(
            1, 2
        ).contiguous()
        attn_output = attn_output.view(
            batch_size, seq_len_q, self.d_model
        )

        # Final linear projection
        output = self.out_proj(attn_output)

        return output


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention module (convenience wrapper).

    Args:
        d_model: The dimension of the model
        num_heads: Number of attention heads
        dropout: Dropout probability (default: 0.1)
        bias: Whether to use bias in linear projections (default: True)
        is_causal: If True, applies causal masking (default: False)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        is_causal: bool = False,
    ):
        super().__init__()
        self.mha = MultiHeadAttention(
            d_model, num_heads, dropout, bias, is_causal
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of self-attention (query, key, value are all the same).

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            attn_mask: Attention mask
            key_padding_mask: Padding mask

        Returns:
            output: Output tensor of shape (batch_size, seq_len, d_model)
        """
        return self.mha(
            x, x, x, attn_mask, key_padding_mask
        )
