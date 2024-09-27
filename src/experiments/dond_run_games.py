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
              player_paths, 
              game_json_path):
    """
    Runs multiple games in parallel and logs the results.

    Args:
        out_dir (str): Output directory for saving game data.
        nb_parallel_games (int): Number of games to run in parallel.
        games_per_iteration (int): Total number of games to run.
        game (DondGame): The game instance.
        players (list): List of player instances.
        models (dict): Dictionary of models to use for generating player moves.
        player_paths (dict): Dictionary of paths to save player contexts.
        game_json_path (str): Path to save game metrics.

    Returns:
        tuple: Paths to player contexts and game JSONs.
    """
    game_nb = 0
    matches = []
    prompt_batches = {model_name: [] for model_name in models.keys()}
    response_batches = {model_name: [] for model_name in models.keys()}

    # Create directories for player contexts and game JSONs
    for path in player_paths.values() + [game_json_path]:
        os.makedirs(path, exist_ok=True)

    nb_matches = min(nb_parallel_games, games_per_iteration)
    for _ in range(nb_matches):
        match = {
            "player_list": [copy.deepcopy(player) for player in players],
            "game": copy.deepcopy(game),
        }
        match["game"].reset()
        match["game_state"] = match["game"].get_state()
        match["play_order"] = match["game"].get_play_order()
        match["player_deque"] = deque(
            [match["player_list"][id] for id in match["play_order"]]
        )
        for i, player in enumerate(match["player_deque"]):
            player.game_id = i
        matches.append(match)

    start_time = time.time()  # Start time for iteration
    logging.info(f"Starting generation of {games_per_iteration} games.")

    while game_nb < games_per_iteration:

        # Get prompt batch for each model
        for match in matches:
            # Add user message to context
            player = match["player_deque"][0]
            match["game_state"] = match["game"].get_state()
            player.set_usr_message(match["game_state"])

            # Send player context to right model
            prompt_batches[player.model_name].append(
                copy.deepcopy(player.get_context())
            )

        # Process prompt batch of each model
        for model_name in models.keys():
            model = models[model_name]
            response_batches[model_name] = model.prompt(prompt_batches[model_name])
            assert len(response_batches[model_name]) == len(prompt_batches[model_name])
            prompt_batches[model_name] = []

        # Play moves for each player by using the model outputs
        for match in matches:
            match["game_state"] = match["game"].get_state()
            player = match["player_deque"][0]
            response = response_batches[player.model_name].pop(0)
            (
                send_to_game,
                is_finalization,
                processed_response,
            ) = player.process_model_response(response, match["game_state"])

            # Player has made an official move (will be other player's turn next)
            if send_to_game:
                match["player_deque"].rotate(1)
                match["game_state"] = match["game"].step(
                    processed_response, is_finalization
                )

                if match["game_state"]["round_ended"]:
                    for player in match["player_list"]:
                        player.set_round_info(match["game_state"])
                    match["play_order"] = match["game"].get_play_order()
                    match["player_deque"] = deque(
                        [match["player_list"][id] for id in match["play_order"]]
                    )
                    for i, player in enumerate(match["player_deque"]):
                        player.game_id = i

                if match["game_state"]["game_ended"]:
                    game_nb += 1
                    log_game(
                        game_nb,
                        match["game"],
                        match["player_deque"],
                        player_paths,
                        game_json_path,
                    )
                    match["game"].reset()
                    for player in match["player_deque"]:
                        player.reset_game()

    end_time = time.time()
    iteration_duration = end_time - start_time
    logging.info(
        f"Generation of {games_per_iteration} games completed in {iteration_duration:.2f} seconds."
    )

    return player_paths, game_json_path

def log_game(game_nb, game, players, player_paths, game_json_path):
    """
    Logs the completion of a game and exports player contexts and game metrics.

    Args:
        game_nb (int): The game number.
        game (DondGame): The game instance.
        players (list): List of player instances.
        player_paths (dict): Dictionary of paths to save player contexts.
        game_json_path (str): Path to save game metrics.
    """
    logging.info(f"Game {game_nb} completed.")

    # Create path
    game_name = f"game_{game_nb:04d}"

    # Export the player contexts
    for player in players:
        player_context_path = os.path.join(
            player_paths[player.player_name], f"{player.player_name}_{game_name}.json"
        )
        os.makedirs(os.path.dirname(player_context_path), exist_ok=True)
        with open(player_context_path, "w") as f:
            json.dump(player.get_context(), f, indent=4)

    # Export game metrics
    rounds_data = game.export()
    df = pd.DataFrame(rounds_data)
    df.set_index("round_id", inplace=True)
    df_transposed = df.transpose()
    game_metrics_path = os.path.join(game_json_path, f"{game_name}.csv")
    os.makedirs(os.path.dirname(game_metrics_path), exist_ok=True)
    df_transposed.to_csv(game_metrics_path)