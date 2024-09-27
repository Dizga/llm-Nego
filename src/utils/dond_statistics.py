import json
import os 
import re
from statistics import mean

def process_game_file(file_path):
    """
    Processes a single game JSON file and returns a dictionary of game statistics.
    """
    # Read the JSON file
    with open(file_path, 'r') as f:
        round_history = json.load(f)

    # Extract necessary data
    quantities = round_history['round_quantities']
    player_0_values = round_history['round_values_player_0']
    player_1_values = round_history['round_values_player_1']
    player_0_finalization = round_history['round_player_0_prop']
    player_1_finalization = round_history['round_player_1_prop']
    agreement_reached = round_history['round_agreement_reached']
    player_0_points = round_history['round_points_player_0']
    player_1_points = round_history['round_points_player_1']

    # Calculate total points for each player
    player_0_total_points = sum(player_0_points)
    player_1_total_points = sum(player_1_points)

    # Calculate total_points_over_maximum for each round
    total_points_over_maximum_list = []
    for i in range(len(quantities)):
        maximum = 0
        p0_points = 0
        p1_points = 0

        if agreement_reached[i]:
            for key in quantities[i]:
                quantity = quantities[i][key]
                max_value = max(player_0_values[i][key], player_1_values[i][key])
                maximum += max_value * quantity
                p0_points += player_0_finalization[i].get(key, 0) * player_0_values[i][key]
                p1_points += player_1_finalization[i].get(key, 0) * player_1_values[i][key]
            total_points = p0_points + p1_points
            total_points_over_maximum = total_points / maximum if maximum > 0 else 0
        else:
            total_points_over_maximum = 0

        total_points_over_maximum_list.append(total_points_over_maximum)

    # Calculate agreement rate and round agreements
    num_rounds = len(quantities)
    agreement_rate = sum(agreement_reached) / num_rounds
    round_agreements = agreement_reached

    # Prepare game statistics
    game_stat = {
        'player_0_total_points': int(player_0_total_points),
        'player_1_total_points': int(player_1_total_points),
        'agreement_rate': agreement_rate,
        'total_points_over_maximum': mean(total_points_over_maximum_list) if total_points_over_maximum_list else 0,
        'round_agreements': round_agreements
    }

    return game_stat

def compute_mean_game_stats(game_stats_list):
    """
    Computes the mean statistics from a list of game statistics.
    """
    if not game_stats_list:
        return {}

    # Initialize accumulators
    total_player_0_points = 0
    total_player_1_points = 0
    total_agreement_rate = 0
    total_points_over_maximum = 0
    num_games = len(game_stats_list)

    # Sum up all the statistics
    for game_stat in game_stats_list:
        total_player_0_points += game_stat['player_0_total_points']
        total_player_1_points += game_stat['player_1_total_points']
        total_agreement_rate += game_stat['agreement_rate']
        total_points_over_maximum += game_stat['total_points_over_maximum']

    # Compute mean statistics
    mean_game_stats = {
        'mean_player_0_total_points': total_player_0_points / num_games,
        'mean_player_1_total_points': total_player_1_points / num_games,
        'mean_agreement_rate': total_agreement_rate / num_games,
        'mean_total_points_over_maximum': total_points_over_maximum / num_games
    }

    return mean_game_stats

def export_dond_statistics(folder_path):
    """
    Computes statistics for all game JSON files in the specified folder.
    Returns a dictionary of mean game statistics.
    """
    game_stats_list = []
    pattern = re.compile(r'^game_\d+\.json$')

    # Process each game file in the folder
    for file_name in os.listdir(folder_path):
        if pattern.match(file_name):
            file_path = os.path.join(folder_path, file_name)
            game_stat = process_game_file(file_path)
            game_stat['file_name'] = file_name  # Optional: include the file name
            game_stats_list.append(game_stat)

    # Save per-game statistics to a JSON file
    game_stats_file = os.path.join(folder_path, 'game_stats.json')
    with open(game_stats_file, 'w') as f:
        json.dump(game_stats_list, f, indent=4)

    # Compute and save mean game statistics
    mean_game_stats = compute_mean_game_stats(game_stats_list)
    mean_game_stats_file = os.path.join(folder_path, 'mean_game_stats.json')
    with open(mean_game_stats_file, 'w') as f:
        json.dump(mean_game_stats, f, indent=4)

    return mean_game_stats