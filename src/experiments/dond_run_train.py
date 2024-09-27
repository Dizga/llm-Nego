import hydra
import os
import logging
import time
from omegaconf import OmegaConf
import random

# Local imports
from src.experiments.dond_run_games import run_games
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

    # Initialize game
    dond_game = DondGame(**cfg["iterations"]["dond_game_args"])

    # Initialize players
    players = [None] * len(cfg["players"].keys())
    for player_name in cfg["players"].keys():
        player_id = cfg["players"][player_name]["id"]
        players[player_id] = DondPlayer(
            **cfg["players"][player_name]["dond_player_args"], player_name=player_name
        )

    player_paths, iteration_folders = initialize_output_paths(cfg, output_directory)

    for iteration in range(cfg["experiment"]["nb_iterations"]):

        # Create / set iteration folders and paths
        it_folder = iteration_folders[iteration]

        # Generate games
        player_paths, games_path = run_games(
            dond_game=dond_game,
            players=players,
            out_paths=player_paths,
            models=models,
            **cfg['run_games_args']
        )

        # Compute iteration statistics
        for player in players:
            player_games_path = player_paths[player.player_name]["game_export_folders"][iteration]
            player_stats_path = player_paths[player.player_name]["local_stat_paths"][iteration]
            player_stats_paths = [path["global_stat_path"] for path in player_paths.values()]
            export_dond_player_stats(player_games_path, player_stats_path)
            export_global_dond_player_stats(player_stats_paths[:iteration+1], 
                                            player_paths[player.player_name]["global_stat_path"])

        # Training models
        for model_name in models.keys():
            model = models[model_name]

            # PPO training
            if model.default_training_mode == "ppo":
                queries, responses, scores = [], [], []

                # Extract data
                for player in players:
                    if player.model_name == model_name:
                        epd_config = cfg["players"][player.player_name]["ppo_data_extraction_args"]
                        player_games_path = player_paths[player.player_name]["game_export_folders"][iteration]
                        new_queries, new_responses, new_scores = extract_ppo_dataset(
                            player_games_path, player.player_name, **epd_config
                        )
                        queries += new_queries
                        responses += new_responses
                        scores += new_scores
                        
                # Shuffle data
                combined = list(zip(queries, responses, scores))
                random.shuffle(combined)
                queries, responses, scores = zip(*combined)

                # Train on data
                it_folder_ppo = os.path.join(it_folder, f"{model_name}_ppo_training")
                export_ppo_training_set(it_folder_ppo, queries, responses, scores)
                model.train_ppo(queries, responses, scores)

            # SFT training
            elif model.default_training_mode == "sft":
                for player in players:
                    file_name = None
                    if player.model_name == model_name:
                        file_name = extract_sft_dataset(it_folder, player.player_name, out_file=file_name)
                model.train_sft(file_name)

    # Calculate and log total duration
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    logging.info(f"Total time taken for the entire run: {total_duration:.2f} seconds")


def initialize_output_paths(cfg, output_directory):
    """
    Initializes all output path names in advance and sets them in a dictionary.

    Args:
        cfg (omegaconf.DictConfig): Configuration object containing all necessary parameters.
        output_directory (str): The base directory for output files.

    Returns:
        tuple: A dictionary containing paths for each player and a list of iteration folders.
    """
    player_paths = {}
    iteration_folders = []

    for iteration in range(cfg["experiment"]["nb_iterations"]):
        it_folder = os.path.join(output_directory, f"iteration_{iteration:04d}")
        iteration_folders.append(it_folder)

    for player_name in cfg["players"].keys():
        player_id = cfg["players"][player_name]["id"]
        global_stat_path = os.path.join(output_directory, f"player_{player_name}_global_stats.json")

        game_export_folders = []
        local_stat_paths = []
        for it_folder in iteration_folders:
            game_export_folder = os.path.join(it_folder, f"player_{player_name}_games")
            local_stat_path = os.path.join(it_folder, f"{player_name}_stats.json")
            game_export_folders.append(game_export_folder)
            local_stat_paths.append(local_stat_path)

        player_paths[player_name] = {
            "global_stat_path": global_stat_path,
            "local_stat_paths": local_stat_paths,
            "game_export_folders": game_export_folders
        }

    return player_paths, iteration_folders