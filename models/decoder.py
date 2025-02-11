# Import libraries and modules
from typing import Tuple, Callable, Optional, Dict
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tensordict import TensorDict

# Custom modules
from models.embeddings import StaticEmbedding
from models.common import ResidualBlock, add_normalization_layer, FP32Attention

@dataclass
class PrecomputedCache:
    init_embeddings: Tensor
    graph_context: Tensor
    glimpse_key: Tensor
    glimpse_val: Tensor
    logit_key: Tensor

class AttentionDecoderWithCache(nn.Module):
    """Attention-based decoder with cache for precomputed values."""
    def __init__(self,
                 action_dim: int,
                 embed_dim: int,
                 seq_dim: int,
                 num_heads: int = 8,
                 dropout_rate: float = 0.1,
                 num_hidden_layers: int = 3,  # Number of hidden layers
                 hidden_dim: int = None,  # Dimension for hidden layers (defaults to 4 * embed_dim)
                 normalization: Optional[str] = None,  # Type of normalization layer
                 init_embedding=None,
                 context_embedding=None,
                 dynamic_embedding=None,
                 temperature: float = 1.0,
                 scale_max: Optional[float] = None,
                 linear_bias: bool = False,
                 max_context_len: int = 256,
                 use_graph_context: bool = False,
                 mask_inner: bool = False,
                 out_bias_pointer_attn: bool = False,
                 check_nan: bool = True,
                 sdpa_fn: Callable = None,
                 **kwargs):
        super(AttentionDecoderWithCache, self).__init__()
        self.context_embedding = context_embedding
        self.dynamic_embedding = dynamic_embedding if dynamic_embedding is not None else StaticEmbedding()
        self.is_dynamic_embedding = not isinstance(self.dynamic_embedding, StaticEmbedding)
        self.action_dim = action_dim
        self.seq_dim = seq_dim
        # Optionally, use graph context
        self.use_graph_context = use_graph_context

        # Configurable Feedforward Network with Variable Hidden Layers
        if hidden_dim is None:
            hidden_dim = 4 * embed_dim  # Default hidden dimension is 4 times the embed_dim

        # Attention Layers
        self.project_embeddings_kv = nn.Linear(embed_dim, embed_dim * 3)  # For key, value, and logit
        self.attention = FP32Attention(embed_dim, num_heads, batch_first=True)
        self.q_norm = add_normalization_layer(normalization, embed_dim)
        self.attn_norm = add_normalization_layer(normalization, embed_dim)

        # Build the layers
        ffn_activation = nn.LeakyReLU()  # nn.GELU(), nn.ReLU(), nn.SiLU(), nn.LeakyReLU()
        norm_fn_input = add_normalization_layer("identity", embed_dim)
        norm_fn_hidden = add_normalization_layer("identity", hidden_dim)
        layers = [
            norm_fn_input,
            nn.Linear(embed_dim, hidden_dim),
            ffn_activation,
        ]
        # Add residual blocks
        for _ in range(num_hidden_layers - 1):
            layers.append(ResidualBlock(hidden_dim, ffn_activation, norm_fn_hidden, dropout_rate,))

        # Output layer
        layers.append(nn.Linear(hidden_dim, embed_dim))
        self.feed_forward = nn.Sequential(*layers)
        self.ffn_norm = add_normalization_layer(normalization, embed_dim)

        # Projection Layers
        self.output_norm = add_normalization_layer(normalization, embed_dim * 2)
        self.mean_head = nn.Linear(embed_dim * 2, action_dim) # Mean head
        self.std_head = nn.Linear(embed_dim * 2, action_dim) # Standard deviation head

        # Temperature for the policy
        self.temperature = temperature
        self.scale_max = scale_max

        # Causal mask to allow anticipating future steps
        self.causal_mask = torch.triu(torch.ones(seq_dim, seq_dim, device='cuda'), diagonal=0)

    def _compute_q(self, cached: PrecomputedCache, td: TensorDict) -> Tensor:
        """Compute query of static and context embedding for the attention mechanism."""
        node_embeds_cache = cached.init_embeddings
        glimpse_q = self.context_embedding(node_embeds_cache, td)
        glimpse_q = glimpse_q.unsqueeze(1) if glimpse_q.ndim == 2 else glimpse_q
        return glimpse_q

    def _compute_kvl(self, cached: PrecomputedCache, td: TensorDict):
        # Compute dynamic embeddings and add to kv embeddings
        node_embeds_cache = cached.init_embeddings
        glimpse_k_stat, glimpse_v_stat, logit_k_stat = (cached.glimpse_key, cached.glimpse_val, cached.logit_key,)
        glimpse_k_dyn, glimpse_v_dyn, logit_k_dyn = self.dynamic_embedding(node_embeds_cache, td)
        glimpse_k = glimpse_k_stat + glimpse_k_dyn
        glimpse_v = glimpse_v_stat + glimpse_v_dyn
        logit_k = logit_k_stat + logit_k_dyn
        return glimpse_k, glimpse_v, logit_k

    def forward(self, td: TensorDict, cached: PrecomputedCache, num_starts: int = 0) -> Tuple[Tensor,Tensor]:
        # Compute query, key, and value for the attention mechanism
        glimpse_q = self._compute_q(cached, td)
        glimpse_q = self.q_norm(glimpse_q)
        glimpse_k, glimpse_v, _ = self._compute_kvl(cached, td)
        # Apply attention mechanism on causal mask to anticipate future steps
        attn_output, _ = self.attention(glimpse_q, glimpse_k, glimpse_v, mask=self.causal_mask)

        # Feedforward Network with Residual Connection block
        attn_output = self.attn_norm(attn_output + glimpse_q)
        ffn_output = self.feed_forward(attn_output)

        # Pointer block to weigh importance of sequence elements
        # The pointer logits (scores) are used to soft select indices over the sequence
        ffn_output = self.ffn_norm(ffn_output + attn_output)
        pointer_logits = torch.matmul(ffn_output, glimpse_k.transpose(-2, -1))  # [batch_size, seq_len, seq_len]
        # Apply the causal mask to pointer logits
        if self.causal_mask is not None:
            causal_mask_t = self.causal_mask[td["timestep"][0],].view(1,-1, self.seq_dim)  # Add batch dimension
            pointer_logits = pointer_logits.masked_fill(causal_mask_t == 0, float('-inf'))

        # Compute the context vector (weighted sum of values based on probabilities)
        pointer_probs = F.softmax(pointer_logits, dim=-1)
        pointer_output = torch.matmul(pointer_probs, glimpse_v)  # [batch_size, seq_len, hidden_dim]

        # Output layer to project pointer_output with ffn_output
        combined_output = torch.cat([ffn_output, pointer_output], dim=-1)
        combined_output = self.output_norm(combined_output)
        # Use mean and std heads for the policy
        mean = F.softplus(self.mean_head(combined_output))
        std = F.softplus(self.std_head(combined_output))

        # Apply temperature scaling and max scaling
        if self.temperature is not None:
            mean = mean/self.temperature
            std = std/self.temperature
        if self.scale_max is not None:
            std = std.clamp(max=self.scale_max)
        return mean.squeeze(), std.squeeze()

    def pre_decoder_hook(self, td: TensorDict, env, embeddings: Tensor, num_starts: int = 0):
        return td, env, self._precompute_cache(embeddings, num_starts)

    def _precompute_cache(self, embeddings: Tensor, num_starts: int = 0) -> PrecomputedCache:
        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = self.project_embeddings_kv(embeddings).chunk(3, dim=-1)

        # Organize in a dataclass for easy access
        return PrecomputedCache(
            init_embeddings=embeddings,
            graph_context=torch.tensor(0),  # Placeholder, can be extended if graph context is used
            glimpse_key=glimpse_key_fixed,
            glimpse_val=glimpse_val_fixed,
            logit_key=logit_key_fixed,
        )

class MLPDecoderWithCache(nn.Module):
    """MLP-based decoder with cache for precomputed values."""
    def __init__(self,
                 action_dim: int,
                 embed_dim: int,
                 seq_dim: int,
                 num_heads: int = 8,
                 dropout_rate: float = 0.1,
                 num_hidden_layers: int = 3,  # Number of hidden layers
                 hidden_dim: int = None,  # Dimension for hidden layers (defaults to 4 * embed_dim)
                 normalization: Optional[str] = None,  # Type of normalization layer
                 obs_embedding=None,
                 temperature: float = 1.0,
                 scale_max: Optional[float] = None,
                 linear_bias: bool = False,
                 max_context_len: int = 256,
                 use_graph_context: bool = False,
                 mask_inner: bool = False,
                 out_bias_pointer_attn: bool = False,
                 check_nan: bool = True,
                 sdpa_fn: Callable = None,
                 **kwargs):
        super(MLPDecoderWithCache, self).__init__()
        self.action_dim = action_dim
        self.obs_embedding = obs_embedding

        # Create policy MLP
        ffn_activation = nn.LeakyReLU()
        norm_fn_input = add_normalization_layer(normalization, embed_dim)
        norm_fn_hidden = add_normalization_layer(normalization, hidden_dim)
        # Build the layers
        layers = [
            norm_fn_input,
            nn.Linear(embed_dim, hidden_dim),
            ffn_activation,
        ]
        # Add residual blocks
        for _ in range(num_hidden_layers - 1):
            layers.append(ResidualBlock(hidden_dim, ffn_activation, norm_fn_hidden, dropout_rate,))

        # Output layer
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.policy_mlp = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.std_head = nn.Linear(hidden_dim, action_dim)

        # Temperature for the policy
        self.temperature = temperature
        self.scale_max = scale_max

    def forward(self, obs, hidden:Optional=None) -> Dict[str, Tensor]:
        # Use the observation embedding to process the input
        hidden = self.obs_embedding(hidden, obs)
        # Compute mask and logits
        hidden = self.policy_mlp(hidden)
        mean = self.mean_head(hidden)
        mean = mean.clamp(min=0.0)
        std = F.softplus(self.std_head(hidden))
        if self.temperature is not None:
            mean = mean/self.temperature
            std = std/self.temperature

        if self.scale_max is not None:
            std = std.clamp(max=self.scale_max)
        return mean, std