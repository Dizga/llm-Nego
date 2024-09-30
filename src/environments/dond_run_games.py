import json
from datetime import datetime
import os
import pandas as pd
import logging
import logging.config
from collections import deque
import copy
import time

# local imports
from environments.dond_game import DondGame
from utils.log_gpu_usage import log_gpu_usage



def run_games(nb_parallel_games, 
              games_per_iteration, 
              game, 
              players, 
              models, 
              player_export_paths, 
              game_json_path=None):
    """
    Runs multiple games in parallel and logs the results.

    Args:
        out_dir (str): Output directory for saving game data.
        nb_parallel_games (int): Number of games to run in parallel.
        games_per_iteration (int): Total number of games to run.
        game (DondGame): The game instance.
        players (dict): Dictionary of players.
        models (dict): Dictionary of models to use for generating player moves.
        player_export_paths (dict): Dictionary of paths to save player contexts.
        game_json_path (str): Path to save game metrics.

    Returns:
        tuple: Paths to player contexts and game JSONs.
    """
    game_nb = 0
    matches = []
    prompt_batches = {model_name: [] for model_name in models.keys()}
    response_batches = {model_name: [] for model_name in models.keys()}


    nb_matches = min(nb_parallel_games, games_per_iteration)

    for _ in range(nb_matches):
        match = {
            "players": {player.player_name: copy.deepcopy(player) for player in players.values()},
            "game": copy.deepcopy(game),
        }
        match["game"].new_game()
        match["game_state"] = match["game"].get_state()
        matches.append(match)

    start_time = time.time()  # Start time for iteration
    logging.info(f"Starting generation of {games_per_iteration} games.")

    while game_nb < games_per_iteration:

        # Get prompt batch for each model
        for match in matches:
            # Add user message to context
            match["game_state"] = match["game"].get_state()
            current_player = match["players"][match["game"].get_current_player()]
            current_player.set_usr_message(match["game_state"])
            prompt_batches[current_player.model_name].append(
                copy.deepcopy(current_player.get_context())
            )

        # Process prompt batch of each model
        for model_name in models.keys():
            model = models[model_name]
            response_batches[model_name] = model.prompt(prompt_batches[model_name])
            prompt_batches[model_name] = []

        # Play moves for each player by using the model outputs
        for match in matches:
            if game_nb  > games_per_iteration:
                break
            match["game_state"] = match["game"].get_state()
            current_player = match["players"][match["game"].get_current_player()]
            response = response_batches[current_player.model_name].pop(0)
            (
                send_to_game,
                is_finalization,
                processed_response,
            ) = current_player.process_model_response(response, match["game_state"])

            # Player has made an official move (will be other player's turn next)
            if send_to_game:
                round_over, game_over, match["game_state"] = match["game"].step(
                    processed_response, is_finalization
                )

                if round_over:
                    for player in match["players"].values():
                        player.set_round_info(match["game_state"])
                        player.new_round()

                if game_over:
                    match["game"].new_game()
                    for player in match["players"].values():
                        player.set_game_info(match["game_state"])
                    log_game(
                        game_nb=game_nb,
                        players=match["players"],
                        player_export_paths=player_export_paths
                    )
                    for player in match["players"].values():
                        player.new_game()
                    game_nb += 1
                    


    end_time = time.time()
    iteration_duration = end_time - start_time
    logging.info(
        f"Generation of {games_per_iteration} games completed in {iteration_duration:.2f} seconds."
    )


def log_game(game_nb=None, game=None, players=None, player_export_paths=None, game_json_path=None):
    """
    Logs the completion of a game and exports player contexts and game metrics.

    Args:
        game_nb (int): The game number.
        game (DondGame): The game instance.
        players (list): List of player instances.
        player_export_paths (dict): Dictionary of paths to save player contexts.
        game_json_path (str): Path to save game metrics.
    """
    logging.info(f"Game {game_nb} completed.")

    # Export the player contexts
    for player_name in players.keys():
        player_save_path = player_export_paths[player_name]
        player_save_path = player_save_path + f"/game_{game_nb:05}.jsonl"
        os.makedirs(os.path.dirname(player_save_path), exist_ok=True)
        with open(player_save_path, "w") as f:
            json.dump(players[player_name].get_augmented_context(), f, indent=4)

    # Export game metrics
    if game_json_path is not None:
        pass

    