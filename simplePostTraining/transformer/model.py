import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .mha import MultiHeadSelfAttention


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Args:
        d_model: Model dimension
        eps: Small value for numerical stability (default: 1e-6)
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of RMSNorm.

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Normalized tensor of shape (..., d_model)
        """
        # Calculate RMS: sqrt(mean(x^2))
        rms = torch.sqrt(
            torch.mean(x**2, dim=-1, keepdim=True)
            + self.eps
        )

        # Normalize and scale
        return x / rms * self.weight


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Feed-Forward Network) for Transformer blocks.

    Args:
        d_model: Input/output dimension
        d_ff: Hidden dimension of the feed-forward network
        dropout: Dropout probability (default: 0.1)
        activation: Activation function (default: 'relu')
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # Linear layers
        self.w1 = nn.Linear(
            d_model, d_ff, bias=False
        )  # Up projection
        self.w2 = nn.Linear(
            d_ff, d_model, bias=False
        )  # Down projection
        self.w3 = nn.Linear(
            d_model, d_ff, bias=False
        )  # Gating (for SwiGLU)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "swiglu":
            self.activation = (
                "swiglu"  # Special case for SwiGLU
            )
        else:
            raise ValueError(
                f"Unsupported activation: {activation}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of MLP.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        if self.activation == "swiglu":
            # SwiGLU activation: Swish(W1(x)) * W3(x)
            return self.w2(
                self.dropout(
                    F.silu(self.w1(x)) * self.w3(x)
                )
            )
        else:
            # Standard activation: W2(activation(W1(x)))
            return self.w2(
                self.dropout(self.activation(self.w1(x)))
            )


class Block(nn.Module):
    """
    Transformer block with multi-head self-attention and MLP.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension (default: 4 * d_model)
        dropout: Dropout probability (default: 0.1)
        activation: MLP activation function (default: 'relu')
        is_causal: Whether to use causal attention (default: False)
        norm_first: Whether to apply RMS norm before attention/MLP (Pre-LN vs Post-LN)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        is_causal: bool = False,
        norm_first: bool = True,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.d_model = d_model
        self.norm_first = norm_first

        # Multi-head self-attention
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            is_causal=is_causal,
        )

        # MLP
        self.mlp = MLP(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
        )

        # RMS normalization
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            attn_mask: Attention mask
            key_padding_mask: Padding mask

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        if self.norm_first:
            # Pre-LN: RMS norm before attention and MLP
            # Self-attention with residual connection
            attn_out = self.attn(
                self.norm1(x), attn_mask, key_padding_mask
            )
            x = x + self.dropout(attn_out)

            # MLP with residual connection
            mlp_out = self.mlp(self.norm2(x))
            x = x + self.dropout(mlp_out)
        else:
            # Post-LN: RMS norm after attention and MLP
            # Self-attention with residual connection
            attn_out = self.attn(
                x, attn_mask, key_padding_mask
            )
            x = x + self.dropout(attn_out)
            x = self.norm1(x)

            # MLP with residual connection
            mlp_out = self.mlp(x)
            x = x + self.dropout(mlp_out)
            x = self.norm2(x)

        return x


class Transformer(nn.Module):
    """
    Full Transformer model with embedding layers and multiple transformer blocks.

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer blocks
        d_ff: Feed-forward hidden dimension (default: 4 * d_model)
        max_seq_len: Maximum sequence length for positional encoding
        dropout: Dropout probability (default: 0.1)
        activation: MLP activation function (default: 'relu')
        is_causal: Whether to use causal attention (default: True)
        norm_first: Whether to apply RMS norm before attention/MLP (default: True)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: Optional[int] = None,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        is_causal: bool = True,
        norm_first: bool = True,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # Token embedding
        self.token_embedding = nn.Embedding(
            vocab_size, d_model
        )

        # Positional encoding
        self.register_buffer(
            "pos_encoding",
            self._create_pos_encoding(max_seq_len, d_model),
            persistent=False,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                Block(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                    is_causal=is_causal,
                    norm_first=norm_first,
                )
                for _ in range(num_layers)
            ]
        )

        # Final layer norm
        self.norm = RMSNorm(d_model)

        # Language modeling head
        self.lm_head = nn.Linear(
            d_model, vocab_size, bias=False
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _create_pos_encoding(
        self, max_seq_len: int, d_model: int
    ) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(
            0, max_seq_len, dtype=torch.float
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_seq_len, d_model)

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of transformer.

        Args:
            input_ids: Token indices of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings
        x = self.token_embedding(
            input_ids
        )  # (batch_size, seq_len, d_model)

        # Add positional encoding
        if seq_len <= self.max_seq_len:
            x = x + self.pos_encoding[:, :seq_len, :].to(
                device
            )
        else:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum sequence length {self.max_seq_len}"
            )

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, key_padding_mask=attention_mask)

        # Final layer norm
        x = self.norm(x)

        # Language modeling head
        logits = self.lm_head(
            x
        )  # (batch_size, seq_len, vocab_size)

        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate text using the transformer.

        Args:
            input_ids: Starting token indices of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens (None = no filtering)
            top_p: Keep tokens with cumulative probability up to p
            attention_mask: Attention mask

        Returns:
            Generated token indices of shape (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        device = input_ids.device
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get logits for the last token
                logits = self.forward(
                    input_ids, attention_mask
                )
                # [batch_size, vocab_size]
                next_token_logits = (
                    logits[:, -1, :] / temperature
                )

                # Apply top-k filtering
                if top_k is not None:
                    top_k = min(
                        top_k, next_token_logits.size(-1)
                    )
                    topk_logits, topk_indices = torch.topk(
                        next_token_logits, top_k
                    )
                    next_token_logits = torch.full_like(
                        next_token_logits, float("-inf")
                    )
                    next_token_logits.scatter_(
                        -1, topk_indices, topk_logits
                    )

                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = (
                        torch.sort(
                            next_token_logits,
                            descending=True,
                        )
                    )
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1),
                        dim=-1,
                    )
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = (
                        cumulative_probs > top_p
                    )
                    sorted_indices_to_remove[..., 1:] = (
                        sorted_indices_to_remove[
                            ..., :-1
                        ].clone()
                    )
                    # always keep the first token
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = (
                        sorted_indices_to_remove.scatter(
                            1,
                            sorted_indices,
                            sorted_indices_to_remove,
                        )
                    )
                    next_token_logits[indices_to_remove] = (
                        float("-inf")
                    )

                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(
                    probs, num_samples=1
                )

                # Append to input_ids
                input_ids = torch.cat(
                    [input_ids, next_token], dim=1
                )

                # Update attention mask if provided
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [
                            attention_mask,
                            torch.ones(
                                attention_mask.shape[0],
                                1,
                                device=device,
                            ),
                        ],
                        dim=1,
                    )

        return input_ids
