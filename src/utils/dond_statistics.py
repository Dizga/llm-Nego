import json
import os
from statistics import mean
from utils.augmented_mean import augmented_mean
from utils.augmented_variance import augmented_variance
from utils.plot_curves import plot_curves
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from torch.utils.tensorboard import SummaryWriter

def process_player_folder(folder_path):
    """
    Processes all player JSON files in the specified folder and returns a list of game statistics.
    """
    player_stats_list = []

    # Process each player file in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as f:
            player_data = json.load(f)
        
        # Assuming each file contains a list of rounds
        for round_data in player_data:
            # Ensure round_data is a dictionary
            if isinstance(round_data, dict):
                game_info = round_data.get('content', round_data)  # Use round_data if 'content' is not present
                player_stats_list.append(game_info)
    
    return player_stats_list

def compute_mean_game_stats(game_stats_list):
    """
    Computes the mean and variance statistics from a list of game statistics.
    """
    if not game_stats_list:
        return {}
    mean_game_stats = {}
    variance_game_stats = {}
    for stat in game_stats_list[0].keys():
        if isinstance(game_stats_list[0][stat], dict): continue
        accumulated_stat = []
        for game_info in game_stats_list:
            accumulated_stat.append(game_info[stat])
        mean_game_stats[stat] = augmented_mean(accumulated_stat)
        variance_game_stats[stat] = augmented_variance(accumulated_stat)
    return mean_game_stats, variance_game_stats

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
    iteration_mean_stats, iteration_variance_stats = compute_mean_game_stats(game_stats_list)

    # Read existing statistics or initialize if file doesn't exist
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            all_stats = json.load(f)
    else:
        all_stats = {"global": {}}

    # Add iteration statistics
    all_stats[f"iteration_{iteration:03d}"] = {
        "mean": iteration_mean_stats,
        "variance": iteration_variance_stats
    }

    # Write updated statistics to file
    with open(output_file, 'w') as f:
        json.dump(all_stats, f, indent=4)

def generate_player_statistics_plots(input_file, output_folder):
    """
    Generates plots for player statistics based on the JSON file and logs them to TensorBoard.
    
    Args:
        input_file (str): Path to the JSON file containing player statistics.
        output_folder (str): Base path to the folder where plots and logs will be saved.
    """
    # Define separate directories for plots and TensorBoard logs
    plot_output_folder = os.path.join(output_folder, "../pngplots")
    tensorboard_output_folder = os.path.join(output_folder, "../tensorboard")

    # Create directories if they don't exist
    os.makedirs(plot_output_folder, exist_ok=True)
    os.makedirs(tensorboard_output_folder, exist_ok=True)

    writer = SummaryWriter(log_dir=tensorboard_output_folder)

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
        for key in iteration_stats[0]['mean'].keys():
            mean_values = [stats['mean'][key] for stats in iteration_stats]
            variance_values = [stats['variance'][key] for stats in iteration_stats]
            iterations = range(1, len(mean_values) + 1)
            
            plt.figure(figsize=(10, 6))

            if isinstance(mean_values[0], list):
                labels = ["Round " + str(i) for i in range(1, len(mean_values[0]) + 1)]
            else:
                labels = ['1']

            # Log to TensorBoard
            for i, mean_value in enumerate(mean_values):
                if isinstance(mean_value, list):
                    for j, val in enumerate(mean_value):
                        writer.add_scalar(f'{key}/Round_{j+1}', val, i+1)
                else:
                    writer.add_scalar(f'{key}', mean_value, i+1)

            #plt.errorbar(iterations, mean_values, yerr=variance_values, label=labels, fmt='-o')
            plt.plot(iterations, mean_values, label=labels)
            plt.xlabel('Iteration')
            plt.ylabel(snake_case_to_title(key))
            plt.title(f"{snake_case_to_title(key)} Through Iterations")
            plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            plt.legend()
            plt.savefig(os.path.join(plot_output_folder, f"{key}_through_iterations.png"), bbox_inches='tight')
            plt.close()

    writer.close()

def snake_case_to_title(snake_str):
    """
    Converts a snake_case string to a title-cased string.
    """
    return ' '.join(word.capitalize() for word in snake_str.split('_'))