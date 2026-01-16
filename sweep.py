import yaml
import wandb
from dotmap import DotMap
from main import main, adapt_env_kwargs
import argparse

if __name__ == "__main__":
    # add command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", nargs="?", default=None, const=None,
                        help="Provide a sweep name to resume an existing sweep, or leave empty to create a new sweep.")
    # Environment parameters
    parser.add_argument('--env_name', type=str, default='block_mpp', help="Name of the environment.")
    parser.add_argument('--ports', type=int, default=4, help="Number of ports in env.")
    parser.add_argument('--bays', type=int, default=20, help="Number of bays in env.")
    parser.add_argument('--capacity', type=list, default=[500], help="Capacity of each bay in TEU.")
    parser.add_argument('--teu', type=int, default=20000, help="Random seed for reproducibility.")
    parser.add_argument('--gen', type=lambda x: x == 'True', default=False)
    parser.add_argument('--ur', type=float, default=1.1)
    parser.add_argument('--cv', type=float, default=0.5)

    # Algorithm parameters
    parser.add_argument('--feasibility_lambda', type=float, default=0.0 #0.2828168389831236
                        , help="Lambda for feasibility.")

    # Model parameters
    parser.add_argument('--encoder_type', type=str, default='attention', help="Type of encoder to use.")
    parser.add_argument('--decoder_type', type=str, default='attention', help="Type of decoder to use.")
    parser.add_argument('--dyn_embed', type=str, default='self_attention', help="Dynamic embedding type.")
    parser.add_argument('--scale_max', type=float, default=100, help="Maximum scale for the model.") # PPO=1.93, SAC=9.46
    parser.add_argument('--projection_type', type=str, default='bound_convex_violation', help="Projection type.")
    parser.add_argument('--projection_kwargs', type=dict, default={'alpha': 0.1, 'delta': 0.1, 'max_iter': 300,
                                                                  'slack_penalty': 1000, 'n_action': 80, 'n_constraints': 85},
                        help="Projection kwargs.")
    parser.add_argument('--block_stowage_mask', type=lambda x: x == 'True', default=True, help="Block stowage mask.")
    parser.add_argument('--use_mask_head', type=bool, default=False, help="Learn mask to optimize paired block stowage.")
    parser.add_argument('--use_preload_mask', type=bool, default=False, help="Use preloaded mask for paired block stowage.")
    parser.add_argument('--normalize_constraints', type=bool, default=False, help="Normalize constraints.")

    # Run parameters
    parser.add_argument('--testing_path', type=str, default='results/trained_models/navigating_uncertainty', help="Path for testing results.")
    parser.add_argument('--phase', type=str, default='train', help="WandB project name.")
    parser.add_argument("--path", type=str, default="results/trained_models/AI2STOW_JOURNAL_VERSION", help="Path to the directory containing the config.yaml and sweep_config.yaml files.")
    parser.add_argument("--folder", type=str, default="sac-vp", help="Folder to save the sweep configuration and results.")
    parser.add_argument('--feasibility_recovery', type=lambda x: x == 'True', default=False, help="Enable feasibility recovery.")
    args = parser.parse_args()

    def train():
        try:
            # Load static configuration from the YAML file
            with open('config.yaml', 'r') as file:
                config = yaml.safe_load(file)
                config = DotMap(config)
                config = adapt_env_kwargs(config)

            # Adjust configuration based on command line arguments
            # Env
            config.env.env_name = args.env_name
            config.env.ports = args.ports
            config.env.TEU = args.teu
            config.env.bays = args.bays
            config.env.capacity = args.capacity
            config.env.generalization = args.gen
            config.env.utilization_rate_initial_demand = args.ur
            config.env.cv_demand = args.cv
            # Algorithm
            config.algorithm.feasibility_lambda = args.feasibility_lambda
            # Model
            config.model.encoder_type = args.encoder_type
            config.model.decoder_type = args.decoder_type
            config.model.dyn_embed = args.dyn_embed
            config.model.scale_max = args.scale_max
            config.training.projection_type = args.projection_type
            config.env.block_stowage_mask = args.block_stowage_mask
            config.model.use_mask_head = args.use_mask_head
            config.model.use_preload_mask = args.use_preload_mask
            config.training.normalize_constraints = args.normalize_constraints

            # Run
            config.testing.folder = args.folder
            config.model.phase = args.phase
            config.testing.feasibility_recovery = args.feasibility_recovery
            n_constraints = config.training.projection_kwargs.n_constraints


            config.algorithm.type, almost_projection_type = config.testing.folder.split("-")
            if almost_projection_type == "vp" or almost_projection_type == "fr+vp":
                config.training.projection_type = "linear_violation"
            elif almost_projection_type == "bvp" or almost_projection_type == "fr+bvp":
                config.training.projection_type = "bound_convex_violation"
            elif almost_projection_type == "ws+pc" or almost_projection_type == "fr+ws+pc":
                config.training.projection_type = "weighted_scaling_policy_clipping"
            elif almost_projection_type == "vp+cp":
                config.training.projection_type = "convex_program"
                config.testing.folder = config.algorithm.type + "-vp"
            elif almost_projection_type == "ws+pc+cp":
                config.training.projection_type = "convex_program"
                config.testing.folder = config.algorithm.type + "-ws+pc"
            elif almost_projection_type == "fr" or almost_projection_type == "pen":
                config.training.projection_type = "None"
            elif almost_projection_type == "pd":
                config.training.projection_type = "None"
                config.algorithm.primal_dual = True
            elif almost_projection_type == "cp":
                config.training.projection_type = "convex_program"
            else:
                raise ValueError(f"Unsupported projection type: {almost_projection_type}")
            print(f"Running with folder: {config.testing.folder}, "
                  f"algorithm type: {config.algorithm.type},"
                  f"generalization: {config.env.generalization},"
                  f"projection type: {config.training.projection_type}")

            # Initialize W&B
            wandb.init(config=config)
            sweep_config = wandb.config

            if almost_projection_type == "pd":
                config['training']['pd_lr'] = sweep_config.pd_lr
                config['algorithm']['feasibility_lambda'] = 1.0
            elif almost_projection_type == "fr":
                config['algorithm']['feasibility_lambda'] = sweep_config.feasibility_lambda
                for i in range(n_constraints):
                    # Error handling for missing lagrangian multipliers
                    if f'lagrangian_multiplier_{i}' not in sweep_config:
                        raise ValueError(f"Missing lagrangian_multiplier_{i} in sweep configuration")
                    config['algorithm'][f'lagrangian_multiplier_{i}'] = sweep_config[f'lagrangian_multiplier_{i}']

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
    with open('sweep_config.yaml') as file:
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
    wandb.agent(sweep_id, function=train, project="mpp_ppo", entity="stowage_planning_research")