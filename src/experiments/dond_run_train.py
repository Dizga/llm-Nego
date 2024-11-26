import hydra
import os
import logging
import time
from omegaconf import OmegaConf
import random
import json
import copy
# Local imports
from models.hf_agent import HfAgent
from environments.dond.dond_player import DondPlayerHandler
from environments.dond.dond_game import DondGame
from models.dummy_hf_agent import DummyHfAgent
from models.oai_agent import OaiAgent
from utils.export_ppo_training_set import export_ppo_training_set
from utils.plot_curves import plot_curves
from utils.dond_statistics import *
from utils.parallel_shuffle import parallel_shuffle
from utils.dond_statistics import update_player_statistics, generate_player_statistics_plots
from training.train_main import *
from generation.run_games import run_matches

def init_models(cfg):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_directory = hydra_cfg["runtime"]["output_dir"]
    os.makedirs(output_directory, exist_ok=True)

    models = {}
    for model_name in cfg["models"].keys():
        if cfg["models"][model_name]["class"] == "hf":
            models[model_name] = HfAgent(**cfg["models"][model_name]["init_args"],
            output_directory=output_directory)
        elif cfg["models"][model_name]["class"] == "dummy_hf":
            models[model_name] = DummyHfAgent(**cfg["models"][model_name]["init_args"])
        elif cfg["models"][model_name]["class"] == "oai":
            models[model_name] = OaiAgent(**cfg["models"][model_name]["init_args"])
    return models


def create_blank_match(
    cfg    
):
    """
    Initializes the matches for the game.

    Args:
        cfg (omegaconf.DictConfig): Configuration object containing all necessary parameters for the negotiation cycle.

    Returns:
        list: A list of match dictionaries.
    """
    players = {}
    for player_name in cfg["matches"]["players"].keys():
        players[player_name] = DondPlayerHandler(player_name, 
        **cfg["matches"]["players"][player_name]["dond_player_args"])
    blank_match = {
        "players": players,
        "game": DondGame(players=list(players.keys()), **cfg["matches"]["dond_game_args"]),
        "game_state": None,
        "stop_condition": cfg["matches"]["stop_condition"],
        "stop_condition_kwargs": cfg["matches"]["stop_condition_kwargs"]
    }
    return blank_match




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
    models = init_models(cfg)

    matches = None

    for iteration in range(cfg["experiment"]["nb_iterations"]):
        iteration_start_time = time.time()

        it_folder = os.path.join(output_directory, f"iteration_{iteration:03}")
        os.makedirs(it_folder, exist_ok=True)


        generation_start_time = time.time()
        matches = [create_blank_match(cfg) for _ in range(cfg["experiment"]["nb_matches_per_iteration"])]
        players = copy.deepcopy(matches[0]["players"])

        run_matches(
            export_path=it_folder,
            matches=matches,
            models=models,
            **cfg['matches']['run_matches_args']
        )

        for player_name in players.keys():
            player_stats_folder = os.path.join(output_directory, "statistics", player_name)
            os.makedirs(player_stats_folder, exist_ok=True)
            player_stats_file = os.path.join(player_stats_folder, f"{player_name}_stats.jsonl")
            player_plots_folder = os.path.join(player_stats_folder, "plots")
            
            update_player_statistics(
                input_path=os.path.join(it_folder, player_name, "statistics"),
                output_file=player_stats_file,
                iteration=iteration
            )
            generate_player_statistics_plots(
                input_file=player_stats_file,
                output_folder=player_plots_folder
            )
        generation_end_time = time.time()

        # Train models
        training_start_time = time.time()


        for model_name, model in models.items():
            for adapter_name in model.adapters.keys():
                mod_adpt_id = f"{model_name}/{adapter_name}"
                model.prepare_adapter_train(adapter_name)

                # Find paths of all player data for this adapter
                data_paths = []
                for player in players.values():
                    if player.mod_adpt_id == mod_adpt_id:
                        player_export_path = os.path.join(it_folder, player.player_name, "training")
                        data_paths.append(player_export_path)

                # Train the adapter by calling train_main with the correct settings
                if data_paths != []:    
                    train_func_args = cfg["training"][model_name]["adapters"][adapter_name]["train_func_args"]
                    train_main(
                        hf_model=model,
                        paths=data_paths,
                        train_func=cfg["training"][model_name]["adapters"][adapter_name]["train_func"],
                        train_func_args=train_func_args,
                        output_path=it_folder
                    )

        training_end_time = time.time()
        iteration_end_time = time.time()

        # Calculate times
        iteration_duration = iteration_end_time - iteration_start_time
        generation_duration = generation_end_time - generation_start_time
        training_duration = training_end_time - training_start_time

        # Calculate percentages
        generation_percentage = (generation_duration / iteration_duration) * 100
        training_percentage = (training_duration / iteration_duration) * 100

        # Estimate remaining time
        elapsed_time = iteration_end_time - total_start_time
        estimated_total_time = iteration_duration * cfg["experiment"]["nb_iterations"]
        estimated_remaining_time = estimated_total_time - elapsed_time # TODO: fix

        # Format time for logging
        def format_time(seconds):
            if seconds >= 3600:
                return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m {int(seconds % 60)}s"
            elif seconds >= 60:
                return f"{int(seconds // 60)}m {int(seconds % 60)}s"
            else:
                return f"{int(seconds)}s"

        logging.info(
            f"Iteration {iteration + 1} took {format_time(iteration_duration)}. "
            f"Generation: {generation_percentage:.2f}%, Training: {training_percentage:.2f}%. "
            f"Estimated time remaining: {format_time(estimated_remaining_time)}. "
            f"Estimated total time for complete run: {format_time(estimated_total_time)}."
        )

    # Calculate and log total duration
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    logging.info(f"Total time taken for the entire run: {format_time(total_duration)}")



