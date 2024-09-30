import hydra
import os
import logging
import time
from omegaconf import OmegaConf
import random
import json

# Local imports
from environments.dond_run_games import run_games
from environments.dond_game import DondGame
from models.hf_agent import HfAgent
from models.dummy_hf_agent import DummyHfAgent
from models.oai_agent import OaiAgent
from environments.dond_player import DondPlayer
from training.extract_ppo_dataset import extract_ppo_dataset
from training.extract_sft_dataset import extract_sft_dataset
from utils.export_ppo_training_set import export_ppo_training_set
from utils.plot_curves import plot_curves
from utils.dond_statistics import export_dond_player_stats, export_global_dond_player_stats
from utils.parallel_shuffle import parallel_shuffle



def dond_run_train(cfg):
    """
    Executes a negotiation cycle for the Deal or No Deal (DoND) game.

    This function initializes models, players, and the game environment based on the provided configuration.
    It then runs multiple iterations where games are generated, statistics are computed, and models are trained
    using either Proximal Policy Optimization (PPO) or Supervised Fine-Tuning (SFT) based on their default training mode.

    Args:
        cfg (omegaconf.DictConfig): Configuration object containing all necessary parameters for the negotiation cycle.
    """
    total_start_time = time.time()

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_directory = hydra_cfg["runtime"]["output_dir"]
    os.makedirs(output_directory, exist_ok=True)

    cfg = OmegaConf.to_container(cfg, resolve=True, structured_config_mode="dict")

    # Initialize models
    models = {}
    for model_name in cfg["models"].keys():
        if cfg["models"][model_name]["class"] == "hf":
            models[model_name] = HfAgent(**cfg["models"][model_name]["init_args"])
        elif cfg["models"][model_name]["class"] == "dummy_hf":
            models[model_name] = DummyHfAgent(**cfg["models"][model_name]["init_args"])
        elif cfg["models"][model_name]["class"] == "oai":
            models[model_name] = OaiAgent(**cfg["models"][model_name]["init_args"])

    # Initialize players
    players = {}
    for player_name in cfg["players"].keys():
        players[player_name] = DondPlayer(player_name, 
        **cfg["players"][player_name]["dond_player_args"])

    # Initialize game
    dond_game = DondGame(players=list(players.keys()), **cfg["dond_game_args"])

    # Initialize output paths
    player_paths = {'player_export_paths': {}, 'local_stat_paths': {}, 'global_stat_paths': {}}
    iteration_folders = {}
    for i in range(cfg["experiment"]["nb_iterations"]):
        it_folder = os.path.join(output_directory, f"iteration_{i:03}")
        iteration_folders[i] = it_folder

        player_paths['player_export_paths'][i] = {player_name: 
                                                it_folder + f"/{player_name}_game_data"
                                                for player_name in players.keys()}
        player_paths['local_stat_paths'][i] = {player_name: 
                                                it_folder + f"/{player_name}_local_stats.json"
                                                for player_name in players.keys()}
        player_paths['global_stat_paths'] = {player_name: 
                                                output_directory + f"/{player_name}_global_stats/"
                                                for player_name in players.keys()}

    for iteration in range(cfg["experiment"]["nb_iterations"]):

        # Create / set iteration folders and paths
        it_folder = iteration_folders[iteration]
        os.makedirs(it_folder, exist_ok=True)

        # Generate games
        run_games(
            game=dond_game,
            players=players,
            player_export_paths=player_paths['player_export_paths'][iteration],
            models=models,
            **cfg['run_games_args']
        )

        # Compute iteration statistics
        for player in players.values():
            export_dond_player_stats(player_paths['player_export_paths'][iteration][player.player_name], 
                                     player_paths["local_stat_paths"][iteration][player.player_name])
            l = [player_paths["local_stat_paths"][i][player.player_name] for i in range(0,iteration+1)]
            export_global_dond_player_stats(l,
                                            player_paths["global_stat_paths"][player.player_name])

        # Training models
        for model_name in models.keys():
            model = models[model_name]

            # PPO training
            if model.default_training_mode == "ppo":
                queries, responses, scores = [], [], []

                # Extract data
                for player in players.values():
                    if player.model_name == model_name:
                        epd_config = cfg["players"][player.player_name]["ppo_data_extraction_args"]
                        player_export_path = player_paths['player_export_paths'][iteration][player.player_name]
                        new_queries, new_responses, new_scores = extract_ppo_dataset(
                            folder_path=player_export_path, **epd_config
                        )
                        queries += new_queries
                        responses += new_responses
                        scores += new_scores
                queries, responses, scores = parallel_shuffle(queries, responses, scores)

                # Train on data
                it_folder_ppo = os.path.join(it_folder, f"{model_name}_ppo_training.jsonl")
                export_ppo_training_set(it_folder_ppo, queries, responses, scores)
                model.train_ppo(queries=queries, responses=responses, scores=scores)

            # SFT training
            elif model.default_training_mode == "sft":
                sft_data = []

                # Extract data
                for player in players.values():
                    if player.model_name == model_name:
                        esd_config = cfg["players"][player.player_name]["sft_data_extraction_args"]
                        player_export_path = player_paths['player_export_paths'][iteration][player.player_name]
                        new_data = extract_sft_dataset(
                            folder_path=player_export_path, **esd_config
                        )
                        sft_data.extend(new_data)

                # Save data to a temporary JSON file
                sft_data_path = os.path.join(it_folder, f"{model_name}_sft_training.jsonl")
                with open(sft_data_path, 'w') as f:
                    json.dump(sft_data, f)

                # Train on data
                model.train_sft(sft_data_path)

                # Remove temporary JSON file
                os.remove(sft_data_path)


    # Calculate and log total duration
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    logging.info(f"Total time taken for the entire run: {total_duration:.2f} seconds")


