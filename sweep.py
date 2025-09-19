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
    parser.add_argument('--block_stowage_mask', type=lambda x: x == 'True', default=True, help="Block stowage mask.")

    # Algorithm parameters
    parser.add_argument('--feasibility_lambda', type=float, default=0.2828168389831236, help="Lambda for feasibility.")

    # Model parameters
    parser.add_argument('--encoder_type', type=str, default='attention', help="Type of encoder to use.")
    parser.add_argument('--decoder_type', type=str, default='attention', help="Type of decoder to use.")
    parser.add_argument('--dyn_embed', type=str, default='self_attention', help="Dynamic embedding type.")
    parser.add_argument('--projection_type', type=str, default='bound_convex_violation', help="Projection type.")
    parser.add_argument('--scale_max', type=float, default=9.46, help="Maximum scale for the model.") # PPO=1.93, SAC=9.46

    # Run parameters
    parser.add_argument('--testing_path', type=str, default='results/trained_models/AI2STOW_JOURNAL_VERSION', help="Path for testing results.")
    parser.add_argument('--phase', type=str, default='train', help="WandB project name.")
    parser.add_argument("--path", type=str, default="results/trained_models/AI2STOW_JOURNAL_VERSION",
                        help="Path to the directory containing the config.yaml and sweep_config.yaml files.")
    parser.add_argument("--folder", type=str, default="sac-bvp",
                        help="Folder to save the sweep configuration and results.")
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
            config.env.block_stowage_mask = args.block_stowage_mask
            # Algorithm
            config.algorithm.feasibility_lambda = args.feasibility_lambda
            # Model
            config.model.encoder_type = args.encoder_type
            config.model.decoder_type = args.decoder_type
            config.model.dyn_embed = args.dyn_embed
            config.model.scale_max = args.scale_max
            config.training.projection_type = args.projection_type
            # Run
            config.testing.folder = args.folder
            config.model.phase = args.phase
            n_constraints = config.training.projection_kwargs.n_constraints

            if config.env.env_name == "mpp":
                # todo: remove?
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
                elif almost_projection_type == "fr":
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

            if config.env.env_name == "mpp":
                # todo: remove?
                if almost_projection_type == "pd":
                    config['training']['pd_lr'] = sweep_config.pd_lr
                    config['algorithm']['feasibility_lambda'] = 1.0
                elif almost_projection_type == "fr":
                    # config['algorithm']['feasibility_lambda'] = sweep_config.feasibility_lambda
                    config['algorithm']['feasibility_lambda'] = 1.0
                    for i in range(n_constraints):
                        config['algorithm'][f'lagrangian_multiplier_{i}'] = sweep_config[f'lagrangian_multiplier_{i}']
                        # Error handling for missing lagrangian multipliers
                        if f'lagrangian_multiplier_{i}' not in sweep_config:
                            raise ValueError(f"Missing lagrangian_multiplier_{i} in sweep configuration")

            # Model hyperparameters
            # config['model']['num_heads'] = sweep_config.num_heads
            # config['model']['dropout_rate'] = sweep_config.dropout_rate
            # config['model']['normalization'] = sweep_config.normalization
            config['model']['hidden_dim'] = sweep_config.hidden_dim
            config['model']['embed_dim'] = sweep_config.embed_dim
            config['model']['num_encoder_layers'] = sweep_config.num_encoder_layers
            config['model']['num_decoder_layers'] = sweep_config.num_decoder_layers
            config['model']['batch_size'] = sweep_config.batch_size
            config['model']['scale_max'] = sweep_config.scale_max
            config['model']['temperature'] = sweep_config.temperature

            # # PPO hyperparameters
            # config['algorithm']['ppo_epochs'] = sweep_config.ppo_epochs
            # config['algorithm']['mini_batch_size'] = sweep_config.mini_batch_size
            # config['algorithm']['entropy_lambda'] = sweep_config.entropy_lambda

            # # AM-PPO hyperparameters
            config['algorithm']['feasibility_lambda'] = sweep_config.feasibility_lambda
            config['training']['lr'] = sweep_config.lr
            config['training']['projection_kwargs']['alpha'] = sweep_config.alpha
            config['training']['projection_kwargs']['delta'] = sweep_config.delta
            config['training']['projection_kwargs']['max_iter'] = sweep_config.max_iter
            # config['training']['projection_kwargs']['scale'] = sweep_config.scale
            # config['training']['pd_lr'] = sweep_config.pd_lr

            # # Algorithm hyperparameters
            # for i in range(n_constraints):
            #     config['algorithm'][f'lagrangian_multiplier_{i}'] = sweep_config[f'lagrangian_multiplier_{i}']
            #     # Error handling for missing lagrangian multipliers
            #     if f'lagrangian_multiplier_{i}' not in sweep_config:
            #         raise ValueError(f"Missing lagrangian_multiplier_{i} in sweep configuration")


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

    # Initialize the sweep with W&B
    if args.sweep:
        sweep_id = args.sweep
    else:
        sweep_id = wandb.sweep(sweep=sweep_config, project="mpp_ppo")

    # Start the sweep agent, which runs the 'train' function with different hyperparameters
    wandb.agent(sweep_id, function=train, project="mpp_ppo", entity="stowage_planning_research")