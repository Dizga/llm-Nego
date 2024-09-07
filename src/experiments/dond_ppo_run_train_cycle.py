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
from models import hf_agent
from src.environments import dond_player

def dond_ppo_run_train_cycle(cfg): 

    # Log total time
    total_start_time = time.time()

    # Make hydra output directory
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_directory = hydra_cfg['runtime']['output_dir']
    os.makedirs(output_directory, exist_ok=True)

    # Get models
    models = {}
    for model_name in cfg.agents.keys():
        models[model_name] = hf_agent(**cfg.agents[model_name])
    
    # Get players
    players = [None*len(cfg.players.keys())]
    for player_name in cfg.players.keys():
        id = cfg.players['player_name'].id
        players[id] = dond_player(**cfg.players[player_name].dond_player_args)
        players[id].model = models[players[player_name].model_name]
        players[id].name = player_name

    # Get game
    dond_game = DondGame(**cfg.iterations.dond_game_args)

    iteration_runner = DondIterationRunner(
        **cfg.iterations.iteration_runner_args, 
        out_dir=output_directory,
        game=dond_game,
        players=players,
        models=models
    )

    for _ in range(cfg.iterations.nb_iterations):
        
        play_start_time = time.time()
        
        # Run iteration
        logging.info(f"Started playing {cfg.playing.games_per_iteration} games.")
        iteration_runner.run_iteration()
        logging.info(f"Completed {cfg.playing.games_per_iteration} games.")
        
        play_end_time = time.time()
        play_duration = play_end_time - play_start_time
        logging.info(f"Time taken for playing {cfg.playing.games_per_iteration} games: {play_duration:.2f} seconds")

        # Get iteration statistics
        compute_dond_statistics(iteration_runner.it_folder)

        train_start_time = time.time()

        logging.info(f"Started {cfg.training.train_type} training.")

        # Train every model on last iteration data
        logging.info(f"Started {cfg.training.train_type} training.")
        for model_name in cfg.agents.keys():
            model_training_data = []
            for player_name in cfg.players.keys():
                if players[player_name].model_name == model_name:
                    training_data += player_training_data 

            models[model_name].train(training_data)

        

        train_end_time = time.time()
        train_duration = train_end_time - train_start_time
        logging.info(f"Time taken for {cfg.training.train_type} training: {train_duration:.2f} seconds")
        logging.info(f"Ended {cfg.training.train_type} training.")
    
    # Total run time
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    logging.info(f"Total time taken for the entire run: {total_duration:.2f} seconds")
