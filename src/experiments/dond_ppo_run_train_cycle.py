import hydra
import os
import os


# local imports
from environments.dond_player import DondPlayer
from utils.dond_logger import DondLogger
from utils.dond_iteration_runner import DondIterationRunner
from environments.dond_game import DondGame
from utils.get_players import setup_players
from utils.train_ppo_agent import train_agent_ppo
from utils.dond_statistics import compute_dond_statistics


def dond_ppo_run_train_cycle(cfg): 
    
    # Make hydra output directory
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_directory = hydra_cfg['runtime']['output_dir']
    os.makedirs(output_directory, exist_ok=True)

    logger = DondLogger(output_directory)

    dond_game = DondGame(**cfg.game)

    players = setup_players(cfg, player_type=DondPlayer)

    iteration_runner = DondIterationRunner(
        cfg.playing.games_per_iteration, 
        game=dond_game,
        players=players,
        logger=logger
    )



    for _ in range(cfg.playing.nb_iterations):
        
        # Play games
        logger.log_info(f"Started playing {cfg.playing.games_per_iteration} games.")
        iteration_runner.run_iteration()
        logger.log_info(f"Completed the {cfg.playing.games_per_iteration} games.")

        # Compute statistics for games played
        compute_dond_statistics(logger.it_folder)

        # Train on games played
        logger.log_info("Started PPO training.")
        train_agent_ppo(agent=players[0].agent, 
                        ppo_trainer_args=cfg.training.ppo_trainer_args, 
                        folder_path=logger.it_folder, 
                        nb_epochs=cfg.training.nb_epochs,
                        logger=logger)
        logger.log_info("Ended PPO training.")



