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
from models.oai_agent import OaiAgent

from environments.dond_player import DondPlayer
from training.extract_ppo_dataset import extract_ppo_dataset
from training.extract_sft_dataset import extract_sft_dataset

def dond_ppo_run_train_cycle(cfg): 
    total_start_time = time.time()

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_directory = hydra_cfg['runtime']['output_dir']
    os.makedirs(output_directory, exist_ok=True)

    cfg = OmegaConf.to_container(cfg, resolve=True, structured_config_mode='dict')

    # Get models
    models = {}
    for model_name in cfg['models'].keys():
        if cfg['models'][model_name]['class'] == "hf":
            models[model_name] = HfAgent(**cfg['models'][model_name]['init_args'], out_dir=output_directory)
            models[model_name].switch_to_generation_mode()
        elif cfg['models'][model_name]['class'] == "dummy_hf":
            models[model_name] = DummyHfAgent(**cfg['models'][model_name]['init_args'])
        elif cfg['models'][model_name]['class'] == "oai":
            models[model_name] = OaiAgent(**cfg['models'][model_name]['init_args'])

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

    iteration_runner = DondIterationRunner(
        **cfg['iterations']['iteration_runner_args'], 
        out_dir=output_directory,
        game=dond_game,
        players=players,
        models=models
    )

    for _ in range(cfg['iterations']['nb_iterations']):
        
        # Generate games
        iteration_runner.run_iteration()
        it_folder = iteration_runner.it_folder

        # Compute iteration statistics
        compute_dond_statistics(it_folder)

        # Training
        for model_name in models.keys():
            model = models[model_name]
            
            model.switch_to_training_mode()

            # Train with ppo
            if model.default_training_mode == 'ppo':
                queries, responses, scores = [], [], []

                for player in players:
                    if player.model_name == model_name:
                        new_queries, new_responses, new_scores = extract_ppo_dataset(it_folder, player.player_name)
                        queries += new_queries
                        responses += new_responses
                        scores += new_scores

                model.train_ppo(queries, responses, scores)

            # Train with supervised fine-tuning
            elif model.default_training_mode == 'sft':
                for player in players:
                    file_name = None
                    if player.model_name == model_name:
                        file_name = extract_sft_dataset(it_folder, player.player_name, out_file=file_name)
                model.train_sft(file_name)

            model.switch_to_generation_mode()
            


    # Calculate and log total duration
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    logging.info(f"Total time taken for the entire run: {total_duration:.2f} seconds")
