import hydra
import os
import logging
import time
from omegaconf import OmegaConf

# local imports
from experiments.dond_iteration_runner import DondIterationRunner
from environments.dond_game import DondGame
from utils.dond_statistics import compute_dond_statistics
from models.hf_agent import HfAgent
from models.dummy_hf_agent import DummyHfAgent

from environments.dond_player import DondPlayer
from training.extract_ppo_dataset import extract_ppo_dataset

def dond_ppo_run_train_cycle(cfg): 
    # Log total time
    total_start_time = time.time()

    # Make hydra output directory
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_directory = hydra_cfg['runtime']['output_dir']
    os.makedirs(output_directory, exist_ok=True)

    # Convert OmegaConf cfg to regular Python dict
    cfg = OmegaConf.to_container(cfg, resolve=True, structured_config_mode='dict')

    # Get models
    models = {}
    for model_name in cfg['models'].keys():
        models[model_name] = HfAgent(**cfg['models'][model_name])
        models[model_name].switch_to_vllm()

    # Get game
    dond_game = DondGame(**cfg['iterations']['dond_game_args'])
    
    # Get players
    players = [None] * len(cfg['players'].keys())
    for player_name in cfg['players'].keys():
        player_id = cfg['players'][player_name]['id']
        players[player_id] = DondPlayer(
            **cfg['players'][player_name]['dond_player_args'], 
            player_name=player_name,
            game_state=dond_game.get_state()
        )

    # Create the iteration runner
    iteration_runner = DondIterationRunner(
        **cfg['iterations']['iteration_runner_args'], 
        out_dir=output_directory,
        game=dond_game,
        players=players,
        models=models
    )

    # Run the iterations
    for _ in range(cfg['iterations']['nb_iterations']):
        
        # Run one iteration
        iteration_runner.run_iteration()
        it_folder = iteration_runner.it_folder

        # Compute iteration statistics
        compute_dond_statistics(it_folder)

        # Train every model on the last iteration's data
        for model_name in models.keys():
            model = models[model_name]
            model.switch_to_hf()

            queries, responses, scores = [], [], []

            # Get training data for the model
            for player in players:
                if player.model_name == model_name:
                    new_queries, new_responses, new_scores = extract_ppo_dataset(it_folder, player.player_name)
                    queries += new_queries
                    responses += new_responses
                    scores += new_scores

            # Train the model using PPO (will automatically save LoRA weights)
            model.train_ppo(it_folder, queries, responses, scores)

            # Go back to generation
            model.switch_to_vllm()

    # Calculate and log total duration
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    logging.info(f"Total time taken for the entire run: {total_duration:.2f} seconds")
