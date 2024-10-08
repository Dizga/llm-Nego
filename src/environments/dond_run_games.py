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


def play_next_move(match, response_batches):
    """
    Plays the next move for the current player in the match.

    Args:
        match (dict): The current match dictionary.
        response_batches (dict): Dictionary of model responses.

    Returns:
        bool: True if the game is over, False otherwise.
    """

    match["game_state"] = match["game"].get_state()
    current_player = match["players"][match["game"].get_current_player()]
    response = response_batches[current_player.model_name].pop(0)
    send_to_game, is_finalization, processed_response = current_player.process_model_response(response, match["game_state"])

    if send_to_game:
        
        round_over, game_over, match["game_state"] = match["game"].step(processed_response, is_finalization)

        if round_over:
            for player in match["players"].values():
                player.set_round_info(match["game_state"])
                player.new_round()

        # Check the stop condition
        if match["stop_condition"](match["game_state"], **match["stop_condition_kwargs"]):
            game_over = True

        if game_over:
            for player in match["players"].values():
                player.set_game_info(match["game_state"])
            return True
        
    return False

def run_matches(
              matches, 
              models, 
              player_export_paths, 
              nb_parallel_matches, 
              game_json_path=None, 
              log_matches=False):  # Add a parameter to control logging
    """
    Runs multiple games in parallel and logs the results.

    Args:
        matches (list): List of match dictionaries.
        models (dict): Dictionary of models to use for generating player moves.
        player_export_paths (dict): Dictionary of paths to save player contexts.
        nb_parallel_matches (int): Number of matches to run in parallel.
        game_json_path (str): Path to save game metrics.
        log_matches (bool): Whether to log matches after completion.

    Returns:
        list: A list of archived games and players.
    """
    match_nb = 0
    all_matches = copy.deepcopy(matches)  # Use the provided list of match dictionaries
    parallel_matches = [all_matches.pop(0) for _ in range(min(nb_parallel_matches, len(all_matches)))]
    archived_games = []

    prompt_batches = {model_name: [] for model_name in models.keys()}
    response_batches = {model_name: [] for model_name in models.keys()}

    start_time = time.time()  # Start time for iteration
    logging.info(f"Starting playing {len(matches)} matches.")

    while parallel_matches or all_matches:

        # Get prompt batch for each model
        for match in parallel_matches:
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
        for match in parallel_matches[:]:  # Iterate over a copy of the list

            if play_next_move(match, response_batches):

                archived_games.append((
                    copy.deepcopy(match["game"]),
                    {name: copy.deepcopy(player) for name, player in match["players"].items()},
                    match_nb
                ))
                match_nb += 1

                # Remove the completed match
                parallel_matches.remove(match)

                # Add a new match from all_matches if available
                if all_matches:
                    parallel_matches.append(all_matches.pop(0))

    # Log matches if the flag is set
    if log_matches:
        for match_nb, (game, players, _) in enumerate(archived_games):
            log_match(
                match_nb=match_nb,
                players=players,
                player_export_paths=player_export_paths
            )

    end_time = time.time()
    iteration_duration = end_time - start_time
    logging.info(
        f"Generation of matches completed in {iteration_duration:.2f} seconds."
    )

    return archived_games

def log_match(match_nb=None, game=None, players=None, player_export_paths=None, game_json_path=None):
    """
    Logs the completion of a match and exports player contexts and game metrics.

    Args:
        match_nb (int): The match number.
        game (DondGame): The game instance.
        players (list): List of player instances.
        player_export_paths (dict): Dictionary of paths to save player contexts.
        game_json_path (str): Path to save game metrics.
    """

    # Export the player contexts
    for player_name in players.keys():
        player_save_path = player_export_paths[player_name]
        player_save_path = player_save_path + f"/game_{match_nb:05}.jsonl"
        os.makedirs(os.path.dirname(player_save_path), exist_ok=True)
        with open(player_save_path, "w") as f:
            json.dump(players[player_name].get_augmented_context(), f, indent=4)

    # Export game metrics
    if game_json_path is not None:
        pass


