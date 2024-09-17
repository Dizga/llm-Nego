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
import copy

def training_tester(cfg): 
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
        if cfg['models'][model_name]['class'] == "hf":
            models[model_name] = HfAgent(**cfg['models'][model_name]['init_args'])
            # models[model_name].switch_to_vllm()

        elif cfg['models'][model_name]['class'] == "dummy_hf":
            models[model_name] = DummyHfAgent(**cfg['models'][model_name]['init_args'])
        elif cfg['models'][model_name]['class'] == "oai":
            models[model_name] = OaiAgent(**cfg['models'][model_name]['init_args'])

    # # Get game
    # dond_game = DondGame(**cfg['iterations']['dond_game_args'])
    
    # # Get players
    # players = [None] * len(cfg['players'].keys())
    # for player_name in cfg['players'].keys():
    #     player_id = cfg['players'][player_name]['id']
    #     players[player_id] = DondPlayer(
    #         **cfg['players'][player_name]['dond_player_args'], 
    #         player_name=player_name,
    #         game_state=dond_game.get_state()
    #     )

    # Create the iteration runner
    # iteration_runner = DondIterationRunner(
    #     **cfg['iterations']['iteration_runner_args'], 
    #     out_dir=output_directory,
    #     game=dond_game,
    #     players=players,
    #     models=models
    # )

    # Run the iterations
    for _ in range(cfg['iterations']['nb_iterations']):
        
        # # Run one iteration
        # iteration_runner.run_iteration()
        # it_folder = iteration_runner.it_folder

        # Compute iteration statistics
        # compute_dond_statistics(it_folder)

        # Train every model on the last iteration's data
        for model_name in models.keys():
            model = models[model_name]
            model.switch_to_hf()

            # Extract data once before training loop
            queries, responses, scores = extract_ppo_dataset(
                '/home/mila/d/dereck.piche/llm-Nego/src/experiments/data', 
                use_pattern_matching=False,
                last_k_responses=1
            )
            good_query = copy.deepcopy(queries[0])

            # TODO: repeat the queries, responses, and scores 100 times (since there are only 2)
            # Filled TODO:
            repeats = 2  # Since there are 2 items, 2 * 50 = 100
            queries *= repeats
            responses *= repeats
            scores *= repeats

            for i in range(2):
                model.train_ppo(
                    '/home/mila/d/dereck.piche/llm-Nego/src/experiments/data', 
                    queries, 
                    responses, 
                    scores
                )
                # TODO: log output of model conditioned on good query
                # Filled TODO:
                output = model.prompt(good_query)
                logging.info(f"Model output after iteration {i}: {output}")


    # Calculate and log total duration
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    logging.info(f"Total time taken for the entire run: {total_duration:.2f} seconds")
