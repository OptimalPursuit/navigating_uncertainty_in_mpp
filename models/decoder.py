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
    latent: Tensor          # [B, N, E]
    graph_context: Tensor            # e.g. [B, E] or placeholder
    glimpse_key: Tensor              # [B, N, E]
    glimpse_val: Tensor              # [B, N, E]
    logit_key: Tensor                # [B, N, E]


class AttentionDecoderWithCache(nn.Module):
    """Attention-based decoder with cache for precomputed values."""
    def __init__(self,
                 action_dim: int,
                 embed_dim: int,
                 seq_dim: int,
                 num_heads: int = 8,
                 dropout_rate: float = 0.1,
                 num_hidden_layers: int = 3,      # Number of hidden layers
                 hidden_dim: int = None,          # Dimension for hidden layers (defaults to 4 * embed_dim)
                 normalization: Optional[str] = None,
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
                 # --- Multi-skip additions ---
                 z_dim: Optional[int] = None,     # dimension of encoder context z; defaults to embed_dim
                 z_skip_site: str = "ffn",        # "ffn" or "attn+ffn"
                 **kwargs):
        super(AttentionDecoderWithCache, self).__init__()

        # Embeddings
        self.context_embedding = context_embedding
        self.dynamic_embedding = dynamic_embedding if dynamic_embedding is not None else StaticEmbedding()
        self.is_dynamic_embedding = not isinstance(self.dynamic_embedding, StaticEmbedding)
        self.action_dim = action_dim
        self.seq_dim = seq_dim

        # Optionally, use graph context
        self.use_graph_context = use_graph_context

        # Configurable Feedforward Network with Variable Hidden Layers
        if hidden_dim is None:
            hidden_dim = 4 * embed_dim

        # Attention Layers
        self.project_embeddings_kv = nn.Linear(embed_dim, embed_dim * 3)  # key, value, logit
        self.attention = FP32Attention(embed_dim, num_heads, batch_first=True)
        self.q_norm = add_normalization_layer(normalization, embed_dim)
        self.attn_norm = add_normalization_layer(normalization, embed_dim)

        # Feed-forward stack
        ffn_activation = nn.LeakyReLU()
        norm_fn_input = add_normalization_layer("identity", embed_dim)
        norm_fn_hidden = add_normalization_layer("identity", hidden_dim)

        layers = [
            norm_fn_input,
            nn.Linear(embed_dim, hidden_dim),
            ffn_activation,
        ]
        for _ in range(num_hidden_layers - 1):
            layers.append(ResidualBlock(hidden_dim, ffn_activation, norm_fn_hidden, dropout_rate))
        layers.append(nn.Linear(hidden_dim, embed_dim))
        self.feed_forward = nn.Sequential(*layers)
        self.ffn_norm = add_normalization_layer(normalization, embed_dim)

        # Output heads
        self.output_norm = add_normalization_layer(normalization, embed_dim * 2)
        self.mean_head = nn.Linear(embed_dim * 2, action_dim)
        self.std_head = nn.Linear(embed_dim * 2, action_dim)

        # Mask head (optional)
        self.use_mask_head = kwargs.get("use_mask_head", False)
        self.use_preload_mask = kwargs.get("use_preload_mask", False)

        if self.use_mask_head:
            self.mask_head = nn.Sequential(
                nn.Linear(embed_dim * 2, hidden_dim),
                add_normalization_layer("identity", hidden_dim),
                nn.LeakyReLU(),
                *[ResidualBlock(hidden_dim, nn.LeakyReLU(), add_normalization_layer("identity", hidden_dim), dropout_rate)
                  for _ in range(num_hidden_layers - 1)],
                nn.Linear(hidden_dim, action_dim)
            )
            self.mask_global_encoder = nn.Sequential(
                nn.Linear(3, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, embed_dim * 2)
            )

        # Policy scaling
        self.temperature = temperature
        self.scale_max = scale_max

        # Causal mask (buffer so it moves with .to(device))
        causal = torch.triu(torch.ones(seq_dim, seq_dim), diagonal=0)
        self.register_buffer("causal_mask", causal, persistent=False)

        # Hyperparameters for mask head
        self.alpha = kwargs.get("alpha", 5.0)
        self.beta = kwargs.get("beta", 1.0)
        self.L = kwargs.get("L", 10.0)
        self.tau_sinkhorn = kwargs.get("tau_sinkhorn", 1.0)
        self.iters_sinkhorn = kwargs.get("iters_sinkhorn", 10)

        self.mask_prob_threshold = kwargs.get("mask_prob_threshold", 0.5)
        self.mask_use_k_pruning = kwargs.get("mask_use_k_pruning", False)
        self.mask_sigmoid_temp = kwargs.get("mask_sigmoid_temp", self.tau_sinkhorn)

        # --- Multi-skip (z injection) ---
        # In your codebase, the most defensible "z" is a global summary of cached.latent (encoder output).
        # If you truly have a separate graph_context vector, you can set use_graph_context=True and provide it properly.
        if z_dim is None:
            z_dim = embed_dim
        self.z_skip_site = z_skip_site  # "ffn" or "attn+ffn"

        self.z_norm = nn.LayerNorm(z_dim)
        # IMPORTANT: project into embed_dim because you add into attn_output / ffn_output which are embed_dim
        self.z_proj = nn.Linear(z_dim, embed_dim, bias=False)

        # Gates are rank-1 (shape [1]) to avoid Kron failing on 0-d params
        self.z_gate_attn = nn.Parameter(torch.zeros(1))
        self.z_gate_ffn = nn.Parameter(torch.zeros(1))

    def _compute_q(self, cached: PrecomputedCache, td: TensorDict) -> Tensor:
        """Compute query from static/context embedding."""
        node_embeds_cache = cached.latent
        glimpse_q = self.context_embedding(node_embeds_cache, td)  # typically [B, E] or [B, 1, E]
        glimpse_q = glimpse_q.unsqueeze(1) if glimpse_q.ndim == 2 else glimpse_q
        return glimpse_q  # [B, N, E]

    def _compute_kvl(self, cached: PrecomputedCache, td: TensorDict) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute K/V/logit keys as static + dynamic."""
        node_embeds_cache = cached.latent
        glimpse_k_stat, glimpse_v_stat, logit_k_stat = (
            cached.glimpse_key, cached.glimpse_val, cached.logit_key,
        )
        glimpse_k_dyn, glimpse_v_dyn, logit_k_dyn = self.dynamic_embedding(node_embeds_cache, td)
        glimpse_k = glimpse_k_stat + glimpse_k_dyn
        glimpse_v = glimpse_v_stat + glimpse_v_dyn
        logit_k = logit_k_stat + logit_k_dyn
        return glimpse_k, glimpse_v, logit_k

    def _gate(self, p: Tensor) -> Tensor:
        """Stable scalar gate in (-1, 1); p is shape [1]."""
        return torch.tanh(p)[0]

    def _get_z_skip(self, cached: PrecomputedCache, N: int) -> Tensor:
        z_pool = cached.latent.mean(dim=1)  # [B, E]
        z_feat = self.z_proj(self.z_norm(z_pool))  # [B, E]
        return z_feat[:, None, :].expand(-1, N, -1)  # [B, Lq, E]

    def forward(self, td: TensorDict, cached: PrecomputedCache, num_starts: int = 0) -> Tuple[Tensor, Tensor, Tensor]:
        # Compute query, key, value
        glimpse_q = self._compute_q(cached, td)             # [B, N, E]
        glimpse_q = self.q_norm(glimpse_q)
        glimpse_k, glimpse_v, _ = self._compute_kvl(cached, td)  # [B, N, E] each
        B, N, E = glimpse_q.shape

        # Multi-skip tensor and gates
        z_skip = self._get_z_skip(cached, N=N)              # [B, N, E]
        g_attn = self._gate(self.z_gate_attn)               # scalar tensor
        g_ffn = self._gate(self.z_gate_ffn)                 # scalar tensor

        # Attention (query length N, key length N)
        attn_output, _ = self.attention(glimpse_q, glimpse_k, glimpse_v, mask=self.causal_mask)

        # Attention residual (+ optional z skip)
        if self.z_skip_site == "attn+ffn":
            attn_output = self.attn_norm(attn_output + glimpse_q + g_attn * z_skip)
        else:
            attn_output = self.attn_norm(attn_output + glimpse_q)

        # FFN
        ffn_core = self.feed_forward(attn_output)

        # FFN residual (+ z skip)
        ffn_output = self.ffn_norm(ffn_core + attn_output + g_ffn * z_skip)

        # Pointer logits: [B, N, N]
        pointer_logits = torch.matmul(ffn_output, glimpse_k.transpose(-2, -1))

        # Apply causal mask (your original indexing assumed timestep indexing into the mask;
        # keep your behavior but ensure shapes match)
        if self.causal_mask is not None:
            # If N == 1, td["timestep"][0] indexing works but produces [1, 1, N] after view/expand.
            causal_mask_t = self.causal_mask[td["timestep"][0],].view(1, -1, self.seq_dim)
            pointer_logits = pointer_logits.masked_fill(causal_mask_t == 0, float("-inf"))

        pointer_probs = F.softmax(pointer_logits, dim=-1)
        pointer_output = torch.matmul(pointer_probs, glimpse_v)  # [B, N, E]

        # Combine and heads
        combined_output = torch.cat([ffn_output, pointer_output], dim=-1)  # [B, N, 2E]
        combined_output = self.output_norm(combined_output)

        mean = F.softplus(self.mean_head(combined_output))
        std = F.softplus(self.std_head(combined_output))

        if self.temperature is not None:
            mean = mean / self.temperature
        if self.scale_max is not None:
            std = std.clamp(max=self.scale_max)

        mask = td.get("action_mask", None)
        if not self.use_mask_head:
            return mean.squeeze(), std.squeeze(), mask

        # If you use mask head, return it too (keep interface consistent)
        # (Replace with your mask-head logic if you have it elsewhere.)
        mask_logits = self.mask_head(combined_output)  # [B, N, action_dim]
        return mean.squeeze(), std.squeeze(), mask_logits.squeeze()

    def pre_decoder_hook(self, td: TensorDict, env, embeddings: Tensor, num_starts: int = 0) -> Tuple[TensorDict, TensorDict, PrecomputedCache]:
        return td, env, self._precompute_cache(embeddings, num_starts)

    def _precompute_cache(self, embeddings: Tensor, num_starts: int = 0) -> PrecomputedCache:
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = self.project_embeddings_kv(embeddings).chunk(3, dim=-1)
        return PrecomputedCache(
            latent=embeddings,
            graph_context=torch.tensor(0, device=embeddings.device),  # placeholder unless you populate it
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

    def forward(self, obs, hidden:Optional=None) -> Tuple[Tensor, Tensor]:
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

def hard_sigmoid_ste(x: Tensor, threshold: float = 0.0, tau: float = 0.1) -> Tensor:
    # soft in (0,1)
    y_soft = torch.sigmoid((x - threshold) / tau)
    # hard in {0,1}
    y_hard = (y_soft > 0.5).to(x.dtype)
    # forward uses hard; backward uses soft gradients
    return y_hard + (y_soft - y_soft.detach())
