import hydra
import os
import logging
import time
from omegaconf import OmegaConf
import random

# Local imports
from src.experiments.dond_run_games import run_games
from environments.dond_game import DondGame
from utils.dond_statistics import compute_dond_statistics
from models.hf_agent import HfAgent
from models.dummy_hf_agent import DummyHfAgent
from models.oai_agent import OaiAgent
from environments.dond_player import DondPlayer
from training.extract_ppo_dataset import extract_ppo_dataset
from training.extract_sft_dataset import extract_sft_dataset
from utils.export_ppo_training_set import export_ppo_training_set
from utils.plot_curves import plot_curves


def dond_nego_cycle(cfg):
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

    for iteration in range(cfg["iterations"]["nb_iterations"]):
        it_folder = os.path.join(output_directory, f"iteration_{iteration:04d}")
        os.makedirs(it_folder, exist_ok=True)

        # Generate games
        player_paths, games_path = run_games(
            it_folder,
            cfg["iterations"]["nb_parallel_games"],
            cfg["iterations"]["games_per_iteration"],
            dond_game,
            players,
            models,
        )

        # Compute iteration statistics
        export_dond_statistics(games_path)
        export_dond_global_statistics(it_folder)

        # Training models
        for model_name in models.keys():
            model = models[model_name]

            # PPO training
            if model.default_training_mode == "ppo":
                queries, responses, scores = [], [], []

                # Extract data
                for player in players:
                    if player.model_name == model_name:
                        epd_config = cfg["players"][player.player_name][
                            "ppo_data_extraction_args"
                        ]
                        player_path = player_paths[player.player_name]
                        (
                            new_queries,
                            new_responses,
                            new_scores,
                        ) = extract_ppo_dataset(
                            player_path, player.player_name, **epd_config
                        )
                        queries += new_queries
                        responses += new_responses
                        scores += new_scores

                # Shuffle data
                # TODO

                # Train on data
                export_ppo_training_set(
                    it_folder + f"/{model_name}_ppo_train_set.jsonl",  # for debugging
                    queries,
                    responses,
                    scores,
                )
                model.train_ppo(queries, responses, scores)

            # SFT training
            elif model.default_training_mode == "sft":
                for player in players:
                    file_name = None
                    if player.model_name == model_name:
                        file_name = extract_sft_dataset(
                            it_folder, player.player_name, out_file=file_name
                        )
                model.train_sft(file_name)

    # Calculate and log total duration
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    logging.info(f"Total time taken for the entire run: {total_duration:.2f} seconds")