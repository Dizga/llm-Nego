import json
import os
from statistics import mean
from utils.augmented_mean import augmented_mean
from utils.plot_curves import plot_curves

def process_player_folder(folder_path):
    """
    Processes all player JSON files in the specified folder and returns a list of game statistics.
    """
    player_stats_list = []

    # Process each player file in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as f: player_data = json.load(f)
        game_info = player_data[0]['content']
        player_stats_list.append(game_info)
    return player_stats_list

def compute_mean_game_stats(game_stats_list):
    """
    Computes the mean statistics from a list of game statistics.
    """
    if not game_stats_list:
        return {}
    mean_game_stats = {}
    for stat in game_stats_list[0].keys():
        if isinstance(game_stats_list[0][stat], dict): continue
        accumulated_stat = []
        for game_info in game_stats_list:
            accumulated_stat.append(game_info[stat])
        mean_game_stats[stat] = augmented_mean(accumulated_stat)
    return mean_game_stats

def export_dond_player_stats(input_path, output_path):
    """
    Computes statistics for all player JSON files in the specified folder.
    Returns a dictionary of mean game statistics.
    """
    game_stats_list = process_player_folder(input_path)
    mean_game_stats = compute_mean_game_stats(game_stats_list)
    with open(output_path, 'w') as f:
        json.dump(mean_game_stats, f, indent=4)

def export_global_dond_player_stats(input_paths, output_path):
    """
    Gathers a list of mean game statistics from multiple folders and generates plots for scalar values.
    """
    all_mean_stats = []

    # Gather mean game statistics from each input path
    for input_path in input_paths:
        mean_game_stats_list = json.load(open(input_path, 'r'))
        all_mean_stats.append(mean_game_stats_list)

    # Transpose the list of dictionaries to a dictionary of lists
    transposed_stats = {}
    for stats in all_mean_stats:
        for key, value in stats.items():
            if key not in transposed_stats:
                transposed_stats[key] = []
            transposed_stats[key].append(value)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Generate plots for scalar values
    for stat, values in transposed_stats.items():
        if all(isinstance(v, (int, float)) for v in values):
            plot_curves(y_list=[values], plot_name=f"{stat}_through_iterations", output_directory=os.path.dirname(output_path))


    # Save the transposed statistics to the output path
    with open(output_path+"/global_stats.json", 'w') as f:
        json.dump(transposed_stats, f, indent=4)
