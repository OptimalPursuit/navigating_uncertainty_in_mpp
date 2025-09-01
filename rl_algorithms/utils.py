from typing import Optional, Dict, Tuple, List, Union
from dotmap import DotMap
from tensordict import TensorDict
import numpy as np
import random
import torch
from environment.env import MasterPlanningEnv, BlockMasterPlanningEnv
from environment.utils import compute_violation

def make_env(env_kwargs:DotMap, batch_size:Union[Tuple,List] = [], device: str = "cuda") -> torch.nn.Module:
    """Setup the environment."""
    if env_kwargs.env_name == "mpp":
        return MasterPlanningEnv(batch_size=batch_size, device=device, **env_kwargs).to(device)
    elif env_kwargs.env_name == "block_mpp":
        return BlockMasterPlanningEnv(batch_size=batch_size, device=device, **env_kwargs).to(device)

def adapt_env_kwargs(config:DotMap) -> DotMap:
    """Adapt environment kwargs based on configuration"""
    if type(config.env.cargo_classes) == DotMap:
        config_env = config.env.value
    else:
        config_env = config.env

    config_env.bays = 10 if config_env.TEU == 1000 else 20
    config_env.weight_classes = 3 if config_env.cargo_classes % 3 == 0 else 2 # 2 weights for 2 classes, 3 weights for 3,6 classes
    config_env.capacity = [50] if config_env.TEU == 1000 else [500]
    config_env.blocks = 1 if config_env.TEU == 1000 else 2
    config_env.block_stowage_mask = False if config_env.TEU == 1000 else True

    # adapt n_action and n_constraints:
    config_env.n_action = 20 if config_env.TEU == 1000 else 80
    config_env.n_constraints = 25 if config_env.TEU == 1000 else 85
    config_env.training.projection_kwargs.n_action = config_env.n_action
    config_env.training.projection_kwargs.n_constraints = config_env.n_constraints
    return config

def recursive_check_for_nans(td:TensorDict, parent_key="") -> None:
    """Recursive check for NaNs and Infs in e.g. TensorDicts and raise an error if found."""
    for key, value in td.items():
        full_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, torch.Tensor):
            check_for_nans(value, full_key)
        elif isinstance(value, TensorDict) or  isinstance(value, Dict):
            # Recursively check nested TensorDicts
            recursive_check_for_nans(value, full_key)

def check_for_nans(tensor, name):
    """Check for NaNs and Infs in a tensor and raise an error if found."""
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        raise ValueError(f"Inf detected in {name}")

def set_unique_seed(batch_index, base_seed=42):
    """Set a unique seed per batch."""
    seed = base_seed + batch_index
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior on CUDA if applicable
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False