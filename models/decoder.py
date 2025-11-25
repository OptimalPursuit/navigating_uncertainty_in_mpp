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
        # do not remove, part of model weights
        self.multiplier_head = nn.Linear(embed_dim * 2, 1) # Multiplier head

        # Mask head: predicts probability of selecting each location (y_head)
        self.use_mask_head = kwargs.get("use_mask_head", False)
        self.use_preload_mask = kwargs.get("use_preload_mask", False)

        if self.use_mask_head:
            # Mask head to predict scores for each location
            self.mask_head = nn.Sequential(
                nn.Linear(embed_dim * 2, hidden_dim),
                add_normalization_layer("identity", hidden_dim),
                nn.LeakyReLU(),
                *[ResidualBlock(hidden_dim, nn.LeakyReLU(), add_normalization_layer("identity", hidden_dim), dropout_rate)
                    for _ in range(num_hidden_layers - 1)
                ],
                nn.Linear(hidden_dim, action_dim)
            )
            # self.mask_head = nn.Sequential(
            #     nn.Linear(embed_dim * 2, hidden_dim),
            #     nn.LayerNorm(hidden_dim),
            #     nn.LeakyReLU(),
            #     nn.Linear(hidden_dim, hidden_dim),
            #     nn.LayerNorm(hidden_dim),
            #     nn.LeakyReLU(),
            #     nn.Linear(hidden_dim, hidden_dim),
            #     nn.LayerNorm(hidden_dim),
            #     nn.LeakyReLU(),
            #     nn.Linear(hidden_dim, action_dim)
            # )

            # Global context encoder for mask prediction
            self.mask_global_encoder = nn.Sequential(
                nn.Linear(3, hidden_dim),  # pod_demand, residual_cap_sum, capacity_to_fill
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, embed_dim * 2)
            )

        # Temperature for the policy
        self.temperature = temperature
        self.scale_max = scale_max

        # Causal mask to allow anticipating future steps
        self.causal_mask = torch.triu(torch.ones(seq_dim, seq_dim, device='cuda'), diagonal=0)

        # Hyperparameters for mask head
        self.alpha = kwargs.get("alpha", 5.0)
        self.beta = kwargs.get("beta", 1.0)
        self.M = kwargs.get("M", 500.0)
        self.tau_sinkhorn = kwargs.get("tau_sinkhorn", 1.0)
        self.iters_sinkhorn = kwargs.get("iters_sinkhorn", 10)

    def _compute_q(self, cached: PrecomputedCache, td: TensorDict) -> Tensor:
        """Compute query of static and context embedding for the attention mechanism."""
        node_embeds_cache = cached.init_embeddings
        glimpse_q = self.context_embedding(node_embeds_cache, td)
        glimpse_q = glimpse_q.unsqueeze(1) if glimpse_q.ndim == 2 else glimpse_q
        return glimpse_q

    def _compute_kvl(self, cached: PrecomputedCache, td: TensorDict) -> Tuple[Tensor, Tensor, Tensor]:
        # Compute dynamic embeddings and add to kv embeddings
        node_embeds_cache = cached.init_embeddings
        glimpse_k_stat, glimpse_v_stat, logit_k_stat = (cached.glimpse_key, cached.glimpse_val, cached.logit_key,)
        glimpse_k_dyn, glimpse_v_dyn, logit_k_dyn = self.dynamic_embedding(node_embeds_cache, td)
        glimpse_k = glimpse_k_stat + glimpse_k_dyn
        glimpse_v = glimpse_v_stat + glimpse_v_dyn
        logit_k = logit_k_stat + logit_k_dyn
        return glimpse_k, glimpse_v, logit_k

    def soft_topk_sinkhorn(self, y_hat, n_needed, tau=1.0, iters=20, eps=1e-6):
        """
        Differentiable soft top-k using Sinkhorn-style normalization.

        Args:
            y_hat: [batch_size, seq_len, n_c] - logits
            n_needed: scalar or [batch_size, seq_len, 1] - approx. number of locations to activate
            tau: temperature for softmax
            iters: number of Sinkhorn iterations
            eps: small number for numerical stability

        Returns:
            mask: [batch_size, seq_len, n_c] - soft mask, sum(mask, dim=-1) ~ n_needed
        """
        B, S, n_c = y_hat.shape

        # Flatten seq dimension for vectorization
        y_flat = y_hat.view(B * S, n_c)

        # Initial softmax (temperature scaling)
        scores = torch.softmax(y_flat / tau, dim=-1)

        # Prepare target sums
        if isinstance(n_needed, (int, float)):
            target_sum = torch.full((B * S, 1), float(n_needed), device=y_hat.device)
        else:
            target_sum = n_needed.view(B * S, 1)

        # Sinkhorn iterations to renormalize to sum ~ n_needed
        for _ in range(iters):
            col_sum = scores.sum(dim=-1, keepdim=True)
            scores = scores * (target_sum / (col_sum + eps))

        # Clamp to [0,1] for safety
        mask_flat = torch.clamp(scores, 0, 1)

        # Reshape back
        mask = mask_flat.view(B, S, n_c)
        return mask

    def soft_topk_batch(self, y_hat, n_needed):
        """
        Vectorized soft-topk over a batch.

        Args:
            y_hat: [batch_size, 1, n_c] - logits for candidate locations
            n_needed: scalar or tensor broadcastable to [batch_size, 1, 1] - approx. number of locations to activate

        Returns:
            mask: [batch_size, 1, n_c] - soft top-k mask
        """
        batch_size, seq_len, n_c = y_hat.shape

        # flatten seq dimension for pairwise differences
        y_flat = y_hat.view(batch_size * seq_len, n_c)  # [B*seq_len, n_c]

        # pairwise differences
        diff = y_flat.unsqueeze(2) - y_flat.unsqueeze(1)  # [B*seq_len, n_c, n_c]
        soft_ranks = torch.sum(torch.sigmoid(diff / self.tau), dim=-1) + 0.5  # [B*seq_len, n_c]

        # broadcast n_needed
        if isinstance(n_needed, (int, float)):
            n_needed = torch.full((batch_size * seq_len, 1), float(n_needed), device=y_hat.device)
        else:
            n_needed = n_needed.view(batch_size * seq_len, 1)

        mask_flat = torch.clamp(n_needed - soft_ranks + 1, 0, 1)  # [B*seq_len, n_c]

        # reshape back to original [batch_size, seq_len, n_c]
        mask = mask_flat.view(batch_size, seq_len, n_c)

        return mask

    def softmin(self, a, b, beta=10.0):
        """Differentiable min approximation."""
        return -1.0 / beta * torch.log(torch.exp(-beta * a) + torch.exp(-beta * b))

    def forward(self, td: TensorDict, cached: PrecomputedCache, num_starts: int = 0) -> Tuple[Tensor,Tensor,Tensor]:
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
            # std = std/self.temperature
        if self.scale_max is not None:
            std = std.clamp(max=self.scale_max)

        # === Predict POD mask based on top-k sinkhorn ===
        if self.use_mask_head:
            # 1. Predict mask logits (learnable)
            y_hat = self.mask_head(combined_output)  # [B, T, A]
            mask_logits = self.alpha * y_hat  # sharpen if needed
            mask_probs = torch.sigmoid(mask_logits)  # learnable soft mask

            # 2. Structured sparsity: soft top-k
            n_needed = td["locations_needed"]
            y_topk = self.soft_topk_sinkhorn(mask_probs, n_needed)  # differentiable sparse mask

            # 3. Apply hard constraints AFTER learning
            if self.use_preload_mask:
                preload_mask = td["preload_mask"].view_as(y_topk)  # binary feasibility mask
                mask_soft = preload_mask * y_topk  # continuous in [0,1]
                mask_hard = preload_mask * (y_topk > 0.5).float()  # discrete {0,1}
            else:
                mask_soft = y_topk  # continuous in [0,1]
                mask_hard = (y_topk > 0.5).float()  # discrete {0,1}

            mask_final = mask_soft + (mask_hard - mask_soft).detach() # STE trick

            # print("--------------------------------")
            # print("y_hat:", y_hat.mean().item())
            # print("mask_logits:", mask_logits.mean().item())
            # print("mask_probs:", mask_probs.mean().item())
            # print("n_needed:", td["locations_needed"].mean().item())
            # print("y_topk:", y_topk.mean(), y_topk.max().item(), y_topk.min().item())
            # print("--------------------------------")
            # print("preload_mask:", preload_mask.sum().item() / preload_mask.numel())
            # print("mask_soft:", mask_soft.sum().item() / mask_soft.numel())
            # print("mask_hard:", mask_hard.sum().item() / mask_hard.numel())
            # print("mask_final:", mask_final.sum().item() / mask_final.numel())

            # 4. DO NOT TOUCH mean or std
            # SAC distribution must be clean and fully learnable
            # mean, std remain unchanged here

            # 5. Return mask for post-sampling application
            # (the caller should do: final_action = action * mask_final)
            return mean.squeeze(), std.clamp(min=1e-3).squeeze(), mask_final.squeeze()
        else:
            # Apply the action mask to the mean and std logits
            mask = td.get("action_mask", None)
            if mask is not None:
                mean = torch.where(mask, mean.squeeze(), 1e-6)
                std = torch.where(mask, std.squeeze(), 1e-6)
            return mean.squeeze(), std.squeeze(), mask

    def pre_decoder_hook(self, td: TensorDict, env, embeddings: Tensor, num_starts: int = 0) -> Tuple[TensorDict, TensorDict, PrecomputedCache]:
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