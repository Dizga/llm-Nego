import json
import os
from statistics import mean
from utils.augmented_mean import augmented_mean
from utils.plot_curves import plot_curves
import matplotlib.pyplot as plt
import numpy as np

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


def update_player_statistics(input_path, output_file, iteration):
    """
    Computes statistics for the current iteration and updates the global statistics file.
    
    Args:
        input_path (str): Path to the folder containing player JSON files for the current iteration.
        output_file (str): Path to the JSON file where statistics are stored.
        iteration (int): Current iteration number.
    """
    game_stats_list = process_player_folder(input_path)
    iteration_stats = compute_mean_game_stats(game_stats_list)

    # Read existing statistics or initialize if file doesn't exist
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            all_stats = json.load(f)
    else:
        all_stats = {"global": {}}

    # Update global statistics
    global_stats = all_stats["global"]
    for key, value in iteration_stats.items():
        if key not in global_stats:
            global_stats[key] = value
        else:
            if isinstance(value, list):
                # For list values, update element-wise
                if not isinstance(global_stats[key], list):
                    global_stats[key] = [global_stats[key]] * len(value)
                global_stats[key] = [(g * iteration + v) / (iteration + 1) 
                                     for g, v in zip(global_stats[key], value)]
            else:
                # For scalar values, update as before
                global_stats[key] = (global_stats[key] * iteration + value) / (iteration + 1)

    # Add iteration statistics
    all_stats[f"iteration_{iteration:03d}"] = iteration_stats

    # Write updated statistics to file
    with open(output_file, 'w') as f:
        json.dump(all_stats, f, indent=4)

def generate_player_statistics_plots(input_file, output_folder):
    """
    Generates plots for player statistics based on the JSON file.
    
    Args:
        input_file (str): Path to the JSON file containing player statistics.
        output_folder (str): Path to the folder where plots will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Read statistics from the JSON file
    try:
        with open(input_file, 'r') as f:
            all_stats = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error reading JSON file: {e}")
        return

    iteration_stats = []
    global_stats = None

    # Process the statistics
    for key, value in all_stats.items():
        if key == 'global':
            global_stats = value
        elif key.startswith('iteration_'):
            iteration_stats.append(value)

    # Generate plots for iteration statistics
    if iteration_stats:
        for key in iteration_stats[0].keys():
            values = [stats[key] for stats in iteration_stats]
            iterations = range(1, len(iteration_stats) + 1)
            
            plt.figure(figsize=(10, 6))
            plt.plot(iterations, values)
            plt.xlabel('Iteration')
            plt.ylabel(key)
            plt.title(f"{key} through iterations")
            plt.savefig(os.path.join(output_folder, f"{key}_through_iterations.png"), bbox_inches='tight')
            plt.close()

    # Generate plots for global statistics
    if global_stats:
        scalar_stats = {}
        non_scalar_stats = {}

        for key, value in global_stats.items():
            if np.isscalar(value):
                scalar_stats[key] = value
            else:
                non_scalar_stats[key] = value

        # Plot scalar statistics
        if scalar_stats:
            plt.figure(figsize=(12, 6))
            keys = list(scalar_stats.keys())
            values = list(scalar_stats.values())
            
            plt.bar(range(len(keys)), values)
            plt.xlabel('Statistic')
            plt.ylabel('Value')
            plt.title('Global Scalar Statistics')
            plt.xticks(range(len(keys)), keys, rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, "global_scalar_statistics.png"), bbox_inches='tight')
            plt.close()

        # Plot non-scalar statistics
        for key, value in non_scalar_stats.items():
            plt.figure(figsize=(10, 6))
            plt.plot(value)
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.title(f'Global Non-Scalar Statistic: {key}')
            plt.savefig(os.path.join(output_folder, f"global_non_scalar_{key}.png"), bbox_inches='tight')
            plt.close()
