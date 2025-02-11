from typing import List, Union
import torch
from torch import nn
from rl4co.models.nn.mlp import MLP
from rl4co.utils.pylogger import get_pylogger
log = get_pylogger(__name__)

class CustomMLP(MLP):
    """Custom MLP model with additional features."""
    @staticmethod
    def _get_norm_layer(norm_method, dim):
        if norm_method == "Batch":
            # Use BatchNorm1d but ensure compatibility with vectorized operations
            in_norm = nn.BatchNorm1d(dim, track_running_stats=False)
        elif norm_method == "Layer":
            # Use LayerNorm
            in_norm = nn.LayerNorm(dim)
        elif norm_method == "None":
            # No normalization
            in_norm = nn.Identity()
        else:
            raise RuntimeError(
                f"Not implemented normalization layer type {norm_method}"
            )
        return in_norm


class ResidualBlock(nn.Module):
    """Residual Block with normalization, activation, and optional dropout."""

    def __init__(self, dim, activation, norm_fn, dropout_rate=None):
        super().__init__()

        # Define layers
        self.linear = nn.Linear(dim, dim)
        self.norm = norm_fn
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate else nn.Identity()

    def forward(self, x):
        residual = x.clone()  # Save residual connection
        x = self.norm(x)  # Apply normalization
        x = self.linear(x)  # Linear transformation
        x = self.activation(x)  # Activation function
        x = self.dropout(x)  # Dropout (if any)
        return x + residual  # Add residual connection

class Permute(nn.Module):
    """Permute layer for reshaping input dimensions."""
    def __init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        if x.dim() == 2:
            return x
        elif x.dim() == 3:
            return x.permute(*self.dims)
        else:
            raise ValueError("Invalid dimensions.")

def add_normalization_layer(normalization, embed_dim):
    """Adds a normalization layer based on the specified type and handles input shape compatibility."""
    if normalization == "batch":
        return nn.Sequential(
            Permute((0, 2, 1)),  # Permute for BatchNorm1d
            nn.BatchNorm1d(embed_dim, track_running_stats=False),
            Permute((0, 2, 1)),  # Revert permutation
        )
    elif normalization == "layer":
        return nn.LayerNorm(embed_dim)
    else:
        return nn.Identity()

class FP32Attention(nn.MultiheadAttention):
    """Multi-head Attention using FP32 computation and FP16 storage, with adjusted initialization."""

    def __init__(self, embed_dim: int, num_heads: int, **kwargs):
        # Ensure embed_dim is divisible by num_heads
        if embed_dim % num_heads != 0:
            print(f"Warning: embed_dim ({embed_dim}) is not divisible by num_heads ({num_heads}). Adjusting embed_dim.")
            embed_dim = (embed_dim // num_heads) * num_heads

        # Call superclass with adjusted embed_dim
        super(FP32Attention, self).__init__(embed_dim, num_heads, **kwargs)
        self.embed_dim = embed_dim  # Store adjusted embed_dim if it was changed
        self.num_heads = num_heads  # Ensure num_heads is consistent

    def forward(self, query, key, value, **kwargs):
        # Cast inputs to FP32 for stable attention computation
        query_fp32 = query.float()
        key_fp32 = key.float()
        value_fp32 = value.float()

        # Ensure head_dim is consistent
        head_dim = self.embed_dim // self.num_heads
        assert self.embed_dim == head_dim * self.num_heads, (
            f"embed_dim ({self.embed_dim}) is not compatible with num_heads ({self.num_heads})."
        )

        # Perform multi-head attention in FP32 and cast back to input dtype
        attn_output_fp32, attn_weights_fp32 = super(FP32Attention, self).forward(query_fp32, key_fp32, value_fp32)
        attn_output = attn_output_fp32.to(query.dtype)
        attn_weights = attn_weights_fp32.to(query.dtype)
        return attn_output, attn_weights