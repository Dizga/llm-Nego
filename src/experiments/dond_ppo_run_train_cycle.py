import hydra
import os
import logging
import logging.config
import time

# local imports
from experiments.dond_iteration_runner import DondIterationRunner
from environments.dond_game import DondGame
from agents.get_dond_players import get_dond_players
from training.train_ppo_agent import train_agent_ppo
from utils.dond_statistics import compute_dond_statistics
from utils.inherit_args import inherit_args


def dond_ppo_run_train_cycle(cfg): 
    # Log total time
    total_start_time = time.time()

    # Make hydra output directory
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_directory = hydra_cfg['runtime']['output_dir']
    os.makedirs(output_directory, exist_ok=True)

    dond_game = DondGame(**cfg.game)
    inherit_args(cfg.player_0, cfg.player_1, "same_as_player_0")
    player_0, player_1 = get_dond_players(dond_game, cfg.player_0, cfg.player_1)

    iteration_runner = DondIterationRunner(
        output_directory,
        cfg.playing.games_per_iteration, 
        game=dond_game,
        players=[player_0, player_1]
    )

    for _ in range(cfg.playing.nb_iterations):
        
        # Time for game generation (playing)
        play_start_time = time.time()
        
        # Play games
        logging.info(f"Started playing {cfg.playing.games_per_iteration} games.")
        iteration_runner.run_iteration()
        logging.info(f"Completed {cfg.playing.games_per_iteration} games.")
        
        play_end_time = time.time()
        play_duration = play_end_time - play_start_time
        logging.info(f"Time taken for playing {cfg.playing.games_per_iteration} games: {play_duration:.2f} seconds")

        compute_dond_statistics(iteration_runner.it_folder)

        # Time for training
        train_start_time = time.time()

        # Train on games played
        logging.info(f"Started {cfg.training.train_type} training.")
        if cfg.training.train_type == "ppo":
            train_agent_ppo(
                agent=player_0.agent, 
                ppo_trainer_args=cfg.training.ppo_trainer_args, 
                folder_path=iteration_runner.it_folder, 
                nb_epochs=cfg.training.nb_epochs
            )
        elif cfg.training.train_type == "bc":
            # TODO: Add behavior cloning training if needed
            pass

        train_end_time = time.time()
        train_duration = train_end_time - train_start_time
        logging.info(f"Time taken for {cfg.training.train_type} training: {train_duration:.2f} seconds")

        if cfg.training.checkpoint_models:  # Export LoRA model weights
            player_0.agent.checkpoint_model(iteration_runner.it_folder)

        logging.info(f"Ended {cfg.training.train_type} training.")
    
    # Total run time
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    logging.info(f"Total time taken for the entire run: {total_duration:.2f} seconds")
