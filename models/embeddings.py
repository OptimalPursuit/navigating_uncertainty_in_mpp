import math
import torch
import torch.nn as nn
from torch import Tensor
from tensordict import TensorDict
from typing import Tuple, Callable, Optional, Dict
from rl4co.utils.ops import gather_by_index

class CargoEmbedding(nn.Module):
    """Cargo embedding of the MPP"""
    def __init__(self, action_dim, embed_dim, seq_dim, env):
        super(CargoEmbedding, self).__init__()
        # Store environment and sequence size
        self.env = env
        self.seq_dim = seq_dim
        self.train_max_demand = self.env.generator.train_max_demand

        # Embedding layers
        if env.name == "port_mpp":
            num_embeddings = self.seq_dim  # Number of embeddings
            self.fc = nn.Linear(num_embeddings, embed_dim)
        else:
            num_embeddings = 7  # Number of embeddings
            self.fc = nn.Linear(num_embeddings, embed_dim)
        self.positional_encoding = DynamicSinusoidalPositionalEncoding(embed_dim)
        self.zeros = torch.zeros(1, seq_dim, embed_dim, device=self.env.device, dtype=self.env.float_type)

    def _combine_cargo_parameters(self, batch_size: Tuple) -> Dict[str, Tensor]:
        """Prepare cargo parameters for init embedding"""
        # Get the batch size
        if batch_size == torch.Size([]):
            norm_features = {
                "pol": (self.env.pol.clone() / self.env.P).view(1, -1),
                "pod": (self.env.pod.clone() / self.env.P).view(1, -1),
                "weights": (self.env.weights[self.env.k].clone() / self.env.weights[self.env.k].max()).view(1, -1),
                "teus": (self.env.teus[self.env.k].clone() / self.env.teus[self.env.k].max()).view(1, -1),
                "revenues": (self.env.revenues.clone() / self.env.revenues.max()).view(1, -1),
            }
        else:
            norm_features = {
                "pol": (self.env.pol.clone() / self.env.P).view(1, -1).expand(batch_size[0], -1),
                "pod": (self.env.pod.clone() / self.env.P).view(1, -1).expand(batch_size[0], -1),
                "weights": (self.env.weights[self.env.k].clone() / self.env.weights[self.env.k].max()).view(1, -1).expand(batch_size[0], -1,),
                "teus": (self.env.teus[self.env.k].clone() / self.env.teus[self.env.k].max()).view(1, -1).expand(batch_size[0], -1,),
                "revenues": (self.env.revenues.clone() / self.env.revenues.max()).view(1, -1).expand(batch_size[0], -1,),
            }
        return norm_features

    def forward(self, td: TensorDict,) -> Tensor:
        cargo_parameters = self._combine_cargo_parameters(batch_size=td.shape)
        max_demand = td["max_demand"]
        if td["expected_demand"].dim() == 2:
            expected_demand = td["expected_demand"] / max_demand
            std_demand = td["std_demand"] / max_demand
        else:
            expected_demand = td["expected_demand"][..., 0, :] / max_demand.squeeze(-1)
            std_demand = td["std_demand"][..., 0, :] / max_demand.squeeze(-1)
        combined_input = torch.stack([expected_demand, std_demand, *cargo_parameters.values()], dim=-1)
        combined_emb = self.fc(combined_input)

        # Positional encoding
        output = self.positional_encoding(combined_emb)
        return output

class CriticEmbedding(nn.Module):
    """Embedding for critic of the MPP"""

    def __init__(self, action_dim, embed_dim, seq_dim, env,):
        super(CriticEmbedding, self).__init__()
        self.env = env
        self.seq_dim = seq_dim
        self.BL = self.env.BL if hasattr(self.env, 'BL') else 1
        self.obs_dim = 1+self.env.T * self.env.K + self.env.B*self.env.D*self.BL + self.env.B-1 + 2 + 3*self.env.B*self.env.D*self.BL
        if not hasattr(self.env, 'BL'):
            self.obs_dim -= 21
        self.project_context = nn.Linear(embed_dim + self.obs_dim, embed_dim,)
        self.train_max_demand = self.env.generator.train_max_demand

    def normalize_obs(self, td:TensorDict) -> Tensor:
        batch_size = td.batch_size
        max_demand = td["max_demand"]
        if hasattr(self.env, 'BL'):
            return torch.cat([
                td["total_profit"] / (td["max_total_profit"]+1e-6),
                (td["observed_demand"] / max_demand ).view(*batch_size, -1),
                (td["residual_capacity"] / self.env.capacity.view(1, -1)).view(*batch_size, -1),
                (td["residual_lc_capacity"] / td["target_long_crane"].unsqueeze(0)).view(*batch_size, -1),
                td["lcg"],
                td["vcg"],
                td["agg_pol_location"] / self.env.P,
                td["agg_pod_location"] / self.env.P,
                td["action_mask"],
            ], dim=-1)
        else:
            return torch.cat([
                (td["observed_demand"] / max_demand ).view(*batch_size, -1),
                (td["residual_capacity"] / self.env.capacity.view(1, -1)).view(*batch_size, -1),
                (td["residual_lc_capacity"] / td["target_long_crane"].unsqueeze(0)).view(*batch_size, -1),
                td["lcg"],
                td["vcg"],
                td["agg_pol_location"] / self.env.P,
                td["agg_pod_location"] / self.env.P,
            ], dim=-1)


    def forward(self,  latent_state: Tensor, td: TensorDict) -> Tensor:
        """Embed the context for the MPP"""
        # Get relevant init embedding
        if td["timestep"].dim() == 1:
            select_init_embedding = gather_by_index(latent_state, td["timestep"][0])
        else:
            select_init_embedding = latent_state.squeeze()

        # Project state, concat embeddings, and project concat to output
        obs = self.normalize_obs(td)
        context_embedding = torch.cat([obs, select_init_embedding], dim=-1)
        output = self.project_context(context_embedding)
        return output


class ContextEmbedding(nn.Module):
    """Context embedding of the MPP"""

    def __init__(self, action_dim, embed_dim, seq_dim, env):
        super(ContextEmbedding, self).__init__()
        self.env = env
        self.seq_dim = seq_dim
        self.BL = self.env.BL if hasattr(self.env, 'BL') else 1
        self.obs_dim = 1+self.env.B*self.env.D*self.BL + self.env.B-1 + 2 + 3*self.env.B*self.env.D*self.BL
        if not hasattr(self.env, 'BL'):
            self.obs_dim -= 21
        self.project_context = nn.Linear(embed_dim + self.obs_dim, embed_dim,)

    def normalize_obs(self, td:TensorDict) -> Tensor:
        batch_size = td.batch_size
        if hasattr(self.env, 'BL'):
            return torch.cat([
                td["total_profit"] / (td["max_total_profit"]+1e-6),
                (td["residual_capacity"] / self.env.capacity.view(1, -1)).view(*batch_size, -1),
                (td["residual_lc_capacity"] / td["target_long_crane"].unsqueeze(0)).view(*batch_size, -1),
                td["lcg"],
                td["vcg"],
                td["agg_pol_location"] / self.env.P,
                td["agg_pod_location"] / self.env.P,
                td["action_mask"],
            ], dim=-1)
        else:
            return torch.cat([
                (td["residual_capacity"] / self.env.capacity.view(1, -1)).view(*batch_size, -1),
                (td["residual_lc_capacity"] / td["target_long_crane"].unsqueeze(0)).view(*batch_size, -1),
                td["lcg"],
                td["vcg"],
                td["agg_pol_location"] / self.env.P,
                td["agg_pod_location"] / self.env.P,
            ], dim=-1)

    def forward(self, latent_state: Tensor, td: TensorDict) -> Tensor:
        """Embed the context for the MPP"""
        # Get relevant init embedding
        if td["timestep"].dim() == 1:
            select_init_embedding = gather_by_index(latent_state, td["timestep"][0])
        else:
            select_init_embedding = latent_state
        # Project state, concat embeddings, and project concat to output
        obs = self.normalize_obs(td)
        context_embedding = torch.cat([obs, select_init_embedding], dim=-1)
        output = self.project_context(context_embedding)
        return output

class DemandSelfAttention(nn.Module):
    """Self-Attention Module for Demand History"""

    def __init__(self, embed_dim, n_heads=4):
        super(DemandSelfAttention, self).__init__()
        self.embed =  nn.Linear(1, embed_dim)  # Projection layer
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, batch_first=True)
        self.fc = nn.Linear(embed_dim, embed_dim)  # Projection layer

    def forward(self, demand: Tensor) -> Tensor:
        """
        Inputs:
            demand_emb: Tensor of shape [batch_size, time_steps, embed_dim]
        Returns:
            attended_demand: Tensor of shape [batch_size, embed_dim]
        """
        demand_emb = self.embed(demand)
        attn_output, _ = self.attention(demand_emb, demand_emb, demand_emb)
        return self.fc(attn_output)  # Aggregate attention-weighted demand


class DynamicSelfAttentionEmbedding(nn.Module):
    """Dynamic embedding with self-attention on demand"""

    def __init__(self, embed_dim, seq_dim, env):
        super(DynamicSelfAttentionEmbedding, self).__init__()
        self.env = env
        self.seq_dim = seq_dim
        self.train_max_demand = self.env.generator.train_max_demand
        self.self_attention = DemandSelfAttention(embed_dim)  # Add self-attention layer
        self.project_dynamic = nn.Linear(2 * embed_dim, 3 * embed_dim)

    def forward(self, latent_state: Optional[Tensor], td: TensorDict) -> Tuple[Tensor, Tensor, Tensor]:
        """Embed the dynamic demand for MPP using self-attention"""
        max_demand = td["max_demand"]
        if td["observed_demand"].dim() == 2:
            observed_demand = td["observed_demand"] / max_demand
        else:
            observed_demand = td["observed_demand"][...,0,:] / max_demand.squeeze(-1)

        # Self-Attention over demand history
        attended_demand = self.self_attention(observed_demand.view(observed_demand.size(0), -1, 1))

        # Combine with latent state
        hidden = torch.cat([attended_demand, latent_state], dim=-1)
        glimpse_k_dyn, glimpse_v_dyn, logit_k_dyn = self.project_dynamic(hidden).chunk(3, dim=-1)
        return glimpse_k_dyn, glimpse_v_dyn, logit_k_dyn

class DynamicEmbedding(nn.Module):
    """Dynamic embedding of the MPP"""

    def __init__(self, embed_dim, seq_dim, env, ):
        super(DynamicEmbedding, self).__init__()
        self.env = env
        self.seq_dim = seq_dim
        self.project_dynamic = nn.Linear(embed_dim + 1, 3 * embed_dim)
        self.train_max_demand = self.env.generator.train_max_demand

    def forward(self, latent_state: Optional[Tensor], td: TensorDict) -> Tuple[Tensor, Tensor, Tensor]:
        """Embed the dynamic demand for the MPP"""
        # Get relevant demand embeddings
        max_demand = td["max_demand"]
        if td["observed_demand"].dim() == 2:
            observed_demand = td["observed_demand"] / max_demand
        else:
            observed_demand = td["observed_demand"][...,0,:] / max_demand

        # Project key, value, and logit to anticipate future steps
        hidden = torch.cat([observed_demand, latent_state], dim=-1)
        glimpse_k_dyn, glimpse_v_dyn, logit_k_dyn = self.project_dynamic(hidden).chunk(3, dim=-1)
        return glimpse_k_dyn, glimpse_v_dyn, logit_k_dyn

class StaticEmbedding(nn.Module):
    """Static embedding as placeholder"""
    # This defines shape of key, value
    def __init__(self, *args, **kwargs):
        super(StaticEmbedding, self).__init__()

    def forward(self, td:TensorDict):
        return 0, 0, 0

class DynamicSinusoidalPositionalEncoding(nn.Module):
    """Dynamic sinusoidal positional encoding"""
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        if x.dim() == 3:
            _, seq_length, _ = x.size()
        else:
            _, seq_length, _, _ = x.size()
        position = torch.arange(seq_length, dtype=torch.float, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2, device=x.device).float() * -(math.log(10000.0) / seq_length))
        pe = torch.zeros(seq_length, self.embed_dim, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return x + pe