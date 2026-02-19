import yaml
import wandb
from dotmap import DotMap
from main import main, adapt_env_kwargs, parse_args
import argparse

if __name__ == "__main__":
    args = parse_args(sweep=True)

    def train():
        try:
            # Load static configuration from the YAML file
            with open('config.yaml', 'r') as file:
                config = yaml.safe_load(file)
                config = DotMap(config)
                config = adapt_env_kwargs(config)

            # Env
            config.env.env_name = args.env_name
            config.env.ports = args.ports
            config.env.TEU = args.teu
            config.env.generalization = args.gen
            config.env.utilization_rate_initial_demand = args.ur
            config.env.cv_demand = args.cv
            config.env.block_stowage_mask = args.block_stowage_mask
            config.env.normalize_constraints = args.normalize_constraints
            # Generator
            config.env.demand_sparsity = args.demand_sparsity
            config.env.demand_perturbation = args.demand_perturbation
            config.env.duration_variable_revenue = args.duration_variable_revenue
            config.env.loading_discharge_region = args.loading_discharge_region
            config.env.use_dirichlet_partition = args.use_dirichlet_partition
            config.env.dirichlet_alpha = args.dirichlet_alpha
            config.env.spot_percentage = args.spot_percentage

            # Algorithm
            config.algorithm.type = args.algorithm_type
            config.algorithm.feasibility_lambda = args.feasibility_lambda
            config.algorithm.primal_dual = args.primal_dual
            # Model
            config.model.encoder_type = args.encoder_type
            config.model.decoder_type = args.decoder_type
            config.model.dyn_embed = args.dyn_embed
            config.model.embed_dim = args.embed_dim
            config.model.hidden_dim = args.hidden_dim
            config.model.temperature = args.temperature
            config.model.scale_max = args.scale_max
            config.model.use_mask_head = args.use_mask_head
            config.model.use_preload_mask = args.use_preload_mask
            # Run
            config.training.learning_rate = args.learning_rate
            config.training.pd_learning_rate = args.pd_learning_rate
            config.training.projection_type = args.projection_type
            config.training.projection_kwargs = DotMap(args.projection_kwargs)
            config.testing.path = args.testing_path
            config.testing.folder = args.folder
            config.model.phase = args.phase
            config.testing.feasibility_recovery = args.feasibility_recovery
            config.testing.num_episodes = args.num_episodes

            if args.feasibility_recovery or (
                    config.training.projection_type == "frank_wolfe" and config.model.phase == "test"):
                config.training.projection_type = "convex_program"
            if config.training.projection_type == "inner_convex_violation_alpha":
                config.training.projection_kwargs.enable_alpha_map = True

            # todo: check the logic of load_config and adapt_env_kwargs, as they are currently called multiple times.
            #  Maybe they can be merged into one function that also handles the command-line arguments?
            config = adapt_env_kwargs(config)

            # # Adjust configuration based on command line arguments
            # # Env
            # config.env.env_name = args.env_name
            # config.env.ports = args.ports
            # config.env.TEU = args.teu
            # config.env.capacity = args.capacity
            # config.env.generalization = args.gen
            # config.env.utilization_rate_initial_demand = args.ur
            # config.env.cv_demand = args.cv
            # # Algorithm
            # config.algorithm.type = args.algorithm_type
            # config.algorithm.feasibility_lambda = args.feasibility_lambda
            # config.algorithm.primal_dual = args.primal_dual
            # # Model
            # config.model.encoder_type = args.encoder_type
            # config.model.decoder_type = args.decoder_type
            # config.model.dyn_embed = args.dyn_embed
            # config.model.scale_max = args.scale_max
            # config.training.projection_type = args.projection_type
            # config.env.block_stowage_mask = args.block_stowage_mask
            # config.model.use_mask_head = args.use_mask_head
            # config.model.use_preload_mask = args.use_preload_mask
            # config.training.normalize_constraints = args.normalize_constraints
            #
            # # Run
            # config.testing.folder = args.folder
            # config.model.phase = args.phase
            # config.testing.feasibility_recovery = args.feasibility_recovery
            # n_constraints = config.training.projection_kwargs.n_constraints
            #
            # config.algorithm.type, almost_projection_type = config.testing.folder.split("-")
            # if almost_projection_type == "vp" or almost_projection_type == "fr+vp":
            #     config.training.projection_type = "linear_violation"
            # elif almost_projection_type == "bvp" or almost_projection_type == "fr+bvp":
            #     config.training.projection_type = "bound_convex_violation"
            # elif almost_projection_type == "ws+pc" or almost_projection_type == "fr+ws+pc":
            #     config.training.projection_type = "weighted_scaling_policy_clipping"
            # elif almost_projection_type == "vp+cp":
            #     config.training.projection_type = "convex_program"
            #     config.testing.folder = config.algorithm.type + "-vp"
            # elif almost_projection_type == "ws+pc+cp":
            #     config.training.projection_type = "convex_program"
            #     config.testing.folder = config.algorithm.type + "-ws+pc"
            # elif almost_projection_type == "fr" or almost_projection_type == "pen":
            #     config.training.projection_type = "None"
            # elif almost_projection_type == "pd":
            #     config.training.projection_type = "None"
            #     config.algorithm.primal_dual = True
            # elif almost_projection_type == "cp":
            #     config.training.projection_type = "convex_program"
            # else:
            #     raise ValueError(f"Unsupported projection type: {almost_projection_type}")
            print(f"Running with folder: {config.testing.folder}, "
                  f"algorithm type: {config.algorithm.type},"
                  f"generalization: {config.env.generalization},"
                  f"projection type: {config.training.projection_type}")

            # Initialize W&B
            wandb.init(config=config)
            sweep_config = wandb.config

            # if almost_projection_type == "pd":
            #     config['training']['pd_lr'] = sweep_config.pd_lr
            #     config['algorithm']['feasibility_lambda'] = 1.0
            # elif almost_projection_type == "fr":
            #     config['algorithm']['feasibility_lambda'] = sweep_config.feasibility_lambda
            #     for i in range(n_constraints):
            #         # Error handling for missing lagrangian multipliers
            #         if f'lagrangian_multiplier_{i}' not in sweep_config:
            #             raise ValueError(f"Missing lagrangian_multiplier_{i} in sweep configuration")
            #         config['algorithm'][f'lagrangian_multiplier_{i}'] = sweep_config[f'lagrangian_multiplier_{i}']

            # Dynamic code to check if keys exist in sweep_config and update config accordingly
            for key in sweep_config.keys():
                if key in config['env']:
                    config['env'][key] = sweep_config[key]
                elif key in config['model']:
                    config['model'][key] = sweep_config[key]
                elif key in config['algorithm']:
                    config['algorithm'][key] = sweep_config[key]
                elif key in config['training']:
                    config['training'][key] = sweep_config[key]

            # Call your main() function
            model = main(config)

            # # Optionally log some results, metrics, or intermediate values here
            # wandb.log({"training_loss": 0.1})  # Example logging
        except Exception as e:
            # Log the error to WandB
            wandb.log({"error": str(e)})

            # Optionally, use WandB alert for critical errors
            wandb.alert(
                title="Training Error",
                text=f"An error occurred during training: {e}",
                level="error"  # 'info' or 'warning' levels can be used as needed
            )

            # Print the error for local console logging as well
            print(f"An error occurred during training: {e}")
        finally:
            wandb.finish()

    # Load the sweep configuration from YAML
    with open('sweep_config_lr.yaml') as file:
        sweep_config = yaml.safe_load(file)

    # Create lagrangian multipliers in sweep_config for each constraint
    if 'default_lagrangian_multiplier' in sweep_config['parameters']:
        n_constraints = args.projection_kwargs['n_constraints']
        for i in range(n_constraints):
            if f'lagrangian_multiplier_{i}' not in sweep_config['parameters']:
                sweep_config['parameters'][f'lagrangian_multiplier_{i}'] = sweep_config['parameters'][f'default_lagrangian_multiplier']

    # Initialize the sweep with W&B
    if args.sweep:
        sweep_id = args.sweep
    else:
        sweep_id = wandb.sweep(sweep=sweep_config, project="mpp_ppo")

    # Start the sweep agent, which runs the 'train' function with different hyperparameters
    wandb.agent(sweep_id, function=train, project="mpp_ppo", entity="stowage_planning_research", count=args.runs_per_agent)