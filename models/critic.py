import copy
from typing import Optional, Union
from tensordict import TensorDict
import torch
from torch import Tensor, nn
from rl4co.utils.pylogger import get_pylogger
log = get_pylogger(__name__)

# Custom imports
from models.common import ResidualBlock, add_normalization_layer

class CriticNetwork(nn.Module):
    """Create a critic network with an encoder and a value head to transform the embeddings to a scalar value."""
    def __init__(
            self,
            encoder: nn.Module,
            value_head: Optional[nn.Module] = None,
            dual_head: Optional[nn.Module] = None,
            primal_dual: Optional[bool] = False,
            n_constraints: int = 25,
            embed_dim: int = 128,
            hidden_dim: int = 512,
            decoder_layers: int = 1,
            context_embedding: Optional[nn.Module] = None,  # Context embedding for additional input
            dynamic_embedding: Optional[nn.Module] = None,  # Dynamic embedding for additional input
            critic_embedding: Optional[nn.Module] = None,  # Observation embedding for additional input
            normalization: Optional[str] = None,
            dropout_rate: Optional[float] = None,
            critic_temperature: float = 1.0,
            customized: bool = False,
            use_q_value: Optional[nn.Module] = False,
            action_dim: Optional[nn.Module] = None,
            **kwargs
    ):
        super(CriticNetwork, self).__init__()

        self.encoder = encoder
        self.context_embedding = context_embedding  # Store context_embedding
        self.dynamic_embedding = dynamic_embedding  # Store dynamic_embedding
        self.critic_embedding = critic_embedding
        self.temperature = critic_temperature
        self.customized = customized
        self.primal_dual = primal_dual
        self.n_constraints = n_constraints

        if use_q_value:
            self.state_action_layer = nn.Linear(embed_dim+action_dim, embed_dim)

        if value_head is None:
            # Adjust embed_dim if encoder has a different dimension
            if getattr(encoder, "embed_dim", embed_dim) != embed_dim:
                log.warning(
                    f"Found encoder with different embed_dim {encoder.embed_dim} than the value head {embed_dim}. "
                    f"Using encoder embed_dim for value head."
                )
                embed_dim = getattr(encoder, "embed_dim", embed_dim)

            # Create value head with residual connections
            ffn_activation = nn.LeakyReLU() # nn.ReLU()
            norm_fn_input = add_normalization_layer(normalization, embed_dim)
            norm_fn_hidden = add_normalization_layer(normalization, hidden_dim)
            # Build the layers
            layers = [
                norm_fn_input,
                nn.Linear(embed_dim, hidden_dim),
                ffn_activation,
            ]
            # Add residual blocks
            for _ in range(decoder_layers - 1):
                layers.append(ResidualBlock(hidden_dim, ffn_activation, norm_fn_hidden, dropout_rate, ))

            # Output layer
            layers.append(nn.Linear(hidden_dim, 1))
            value_head = nn.Sequential(*layers)

        self.value_head = value_head

        if self.primal_dual and dual_head == None:
            # Create dual head with residual connections
            layers = [
                add_normalization_layer(normalization, embed_dim),
                nn.Linear(embed_dim, hidden_dim),
                nn.LeakyReLU(),  # nn.ReLU()
            ]
            for _ in range(decoder_layers - 1):
                layers.append(ResidualBlock(hidden_dim, nn.LeakyReLU(), add_normalization_layer(normalization, hidden_dim), dropout_rate, ))

            layers.append(nn.Linear(hidden_dim, n_constraints))
            layers.append(LogUniformActivation(min_val=0.01, max_val=10.0))  # Log uniform activation
            dual_head = nn.Sequential(*layers)

        self.dual_head = dual_head

    def forward(self, obs: Union[Tensor, TensorDict], action: Optional[Tensor] = None) -> Union[Tensor, tuple]:
        # Encode the input
        encoded_h, _ = self.encoder(obs)  # [batch_size, N, embed_dim] -> [batch_size, N]
        embedded_h = self.critic_embedding(encoded_h, obs)

        # State-action value
        if action is not None and hasattr(self, "state_action_layer"):
            if embedded_h.size(0) != action.size(0):
                raise ValueError("Batch size of `embedded_h` and `action` must match.")
            concatenated_h = torch.cat([embedded_h, action.clone()], dim=-1)
            state_action_h = self.state_action_layer(concatenated_h)
        else:
            state_action_h = embedded_h

        # Compute the value
        if not self.customized:  # for most constructive tasks
            output = self.value_head(state_action_h).sum(dim=1, keepdims=True)  # [batch_size, N] -> [batch_size, 1]
        else:  # customized encoder and value head with hidden input
            output = self.value_head(state_action_h)  # [batch_size, N] -> [batch_size, N]
        output = output / self.temperature  # Safe division

        # Compute the Lagrangian multiplier
        if self.primal_dual:
            lagrangian_multiplier = self.dual_head(state_action_h)
            return output, lagrangian_multiplier
        else:
            return output  # ensures [B, S] if it was [B, S, 1]

def create_critic_from_actor(
    policy: nn.Module, backbone: str = "encoder", **critic_kwargs
) -> nn.Module:
    # we reuse the network of the policy's backbone, such as an encoder
    encoder = getattr(policy, backbone, None)
    if encoder is None:
        raise ValueError(
            f"CriticBaseline requires a backbone in the policy network: {backbone}"
        )
    critic = CriticNetwork(copy.deepcopy(encoder), **critic_kwargs).to(
        next(policy.parameters()).device
    )
    return critic

class LogUniformActivation(nn.Module):
    def __init__(self, min_val=0.01, max_val=10.0):
        super().__init__()
        self.min_val = min_val
        self.log_scale = torch.log(torch.tensor(max_val / min_val))

    def forward(self, x):
        return self.min_val * torch.exp(torch.sigmoid(x) * self.log_scale)
