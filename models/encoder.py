from typing import Tuple, Union, Optional
from tensordict import TensorDict
from torch import Tensor
import torch.nn as nn

# Custom
from models.common import CustomMLP

class MLPEncoder(nn.Module):
    """MLP Encoder.
    First embed the input and then process it with a feedforward network.

    Args:
        embed_dim: Dimension of the embedding space
        init_embedding: Module to use for the initialization of the embeddings
        env_name: Name of the environment used to initialize embeddings
        num_heads: Number of heads in the attention layers
        num_layers: Number of layers in the attention network
        normalization: Normalization type in the attention layers
        feedforward_hidden: Hidden dimension in the feedforward layers
        net: Graph Attention Network to use
        sdpa_fn: Function to use for the scaled dot product attention
        moe_kwargs: Keyword arguments for MoE
    """

    def __init__(
        self,
        embed_dim: int = 128,
        init_embedding: nn.Module = None,
        env_name: str = "mpp",
        num_heads: int = 8,
        num_layers: int = 3,
        normalization: str = "layer",
        feedforward_hidden: int = 512,
        net: nn.Module = None,
        sdpa_fn = None,
        moe_kwargs: dict = None,
    ):
        super(MLPEncoder, self).__init__()
        self.env_name = env_name
        self.init_embedding = init_embedding
        normalization = "Batch" if normalization == "batch" else normalization
        normalization = "Layer" if normalization == "layer" else normalization

        self.net = (
            CustomMLP(
                embed_dim,
                embed_dim,
                [feedforward_hidden] * num_layers,
                hidden_act="ReLU",
                out_act="Identity",
                input_norm=normalization,
                output_norm=normalization,
            )
        )

    def forward(
        self, td: TensorDict, mask: Union[Tensor, None] = None
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass of the encoder.
        Transform the input TensorDict into a latent representation.

        Args:
            td: Input TensorDict containing the environment state
            mask: Mask to apply to the attention

        Returns:
            h: Latent representation of the input
            init_h: Initial embedding of the input
        """
        # Transfer to embedding space
        batch_size = td.batch_size
        init_h = self.init_embedding(td)

        # Process embedding
        h = self.net(init_h.view(-1, init_h.size(-1)))

        # Return latent representation and initial embedding
        return h.view(*batch_size, -1, h.size(-1)), init_h.view(*batch_size, -1, init_h.size(-1))

class AttentionEncoder(nn.Module):
    """Graph Attention Encoder as in Kool et al. (2019).
    First embed the input and then process it with a Graph Attention Network.

    Args:
        embed_dim: Dimension of the embedding space
        init_embedding: Module to use for the initialization of the embeddings
        env_name: Name of the environment used to initialize embeddings
        num_heads: Number of heads in the attention layers
        num_layers: Number of layers in the attention network
        normalization: Normalization type in the attention layers
        feedforward_hidden: Hidden dimension in the feedforward layers
        net: Graph Attention Network to use
        sdpa_fn: Function to use for the scaled dot product attention
        moe_kwargs: Keyword arguments for MoE
    """

    def __init__(
        self,
        embed_dim: int = 128,
        init_embedding: nn.Module = None,
        env_name: str = "mpp",
        num_heads: int = 8,
        num_layers: int = 3,
        normalization: str = "layer",
        feedforward_hidden: int = 512,
        net: nn.Module = None,
        sdpa_fn = None,
        moe_kwargs: dict = None,
        **kwargs,
    ):
        super(AttentionEncoder, self).__init__()
        self.env_name = env_name
        self.init_embedding = init_embedding
        self.net = (
            GraphAttentionNetwork(
                num_heads,
                embed_dim,
                num_layers,
                normalization,
                feedforward_hidden,
                sdpa_fn=sdpa_fn,
                moe_kwargs=moe_kwargs,
            )
            if net is None
            else net
        )

    def forward(
        self, td: TensorDict, mask: Union[Tensor, None] = None
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass of the encoder.
        Transform the input TensorDict into a latent representation.

        Args:
            td: Input TensorDict containing the environment state
            mask: Mask to apply to the attention

        Returns:
            h: Latent representation of the input
            init_h: Initial embedding of the input
        """
        # Transfer to embedding space
        init_h = self.init_embedding(td)

        # Process embedding
        h = self.net(init_h, mask)

        # Return latent representation and initial embedding
        return h, init_h

class GraphAttentionNetwork(nn.Module):
    """Manually implemented Graph Attention Network with multiple MHA layers."""

    def __init__(
            self,
            num_heads: int,
            embed_dim: int,
            num_layers: int,
            normalization: str = "batch",
            feedforward_hidden: int = 512,
            **kwargs,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            MultiHeadAttentionLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                feedforward_hidden=feedforward_hidden,
                normalization=normalization,
                **kwargs,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass of the encoder."""
        assert mask is None, "Mask not yet supported!"

        for layer in self.layers:
            x = layer(x)

        return x


class MultiHeadAttentionLayer(nn.Module):
    """Manually implemented Multi-Head Attention Layer with normalization and feed-forward."""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int = 8,
            feedforward_hidden: int = 512,
            normalization: Optional[str] = "batch",
            bias: bool = True,
            **kwargs,
    ):
        super().__init__()

        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, bias=bias)
        self.norm1 = nn.BatchNorm1d(embed_dim) if normalization == "batch" else nn.LayerNorm(embed_dim)
        self.norm2 = nn.BatchNorm1d(embed_dim) if normalization == "batch" else nn.LayerNorm(embed_dim)

        # Define feed-forward network
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_hidden),
            nn.ReLU(),
            nn.Linear(feedforward_hidden, embed_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        # Multi-Head Attention with residual connection
        attn_output, _ = self.mha(x, x, x)
        x = x + attn_output
        if isinstance(self.norm1, nn.BatchNorm1d):
            x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)  # BatchNorm expects (N, C, L)
        else:
            x = self.norm1(x)

        # Feed-forward network with residual connection
        ff_output = self.feedforward(x)
        x = x + ff_output
        if isinstance(self.norm2, nn.BatchNorm1d):
            x = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            x = self.norm2(x)

        return x