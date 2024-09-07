import hydra
import os
import logging
import logging.config
import time

# local imports
from experiments.dond_iteration_runner import DondIterationRunner
from environments.dond_game import DondGame
from utils.dond_statistics import compute_dond_statistics
from utils.inherit_args import inherit_args
from models import hf_agent
from src.environments import dond_player
from src.utils import extract_dond_ppo_dataset

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
        models[model_name].switch_to_vllm()
    
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
        
        it_folder = iteration_runner.it_folder
        
        # Run iteration
        iteration_runner.run_iteration()

        # Get iteration statistics
        compute_dond_statistics(it_folder)

        # Train every model on last iteration data
        for model_name in cfg.agents.keys():

            model = models[model_name]
            model.switch_to_hf()
            queries, responses, scores = [], [], []

            for player_name in cfg.players.keys():

                if players[player_name].model_name == model_name:
                    new_queries, new_responses, new_scores = extract_dond_ppo_dataset(it_folder, player_name)
                    queries += new_queries; responses += new_responses; new_scores += new_scores

            model.train_ppo(queries, responses, scores)

            if model.save_lora_weights: 
                model.save_lora_weights(os.path.join(it_folder, model_name+'_lora_weights'))

   
    # Total run time
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    logging.info(f"Total time taken for the entire run: {total_duration:.2f} seconds")
