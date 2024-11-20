import os
import json
from .dond_statistics_funcs import gather_dond_statistics
from .dond_return_funcs import set_discounted_returns

def independant_players_logging(
        path,
        player_infos, 
        info,
        training_data_func,
        training_data_func_args,
        metrics_func,
        metrics_func_args
        ):
    """
    Logs the training data and metrics independently for each player in a match.
    """
    for player_info in player_infos:
        player_name = player_info["player_name"]
        
        # Define paths for training and statistics subfolders
        training_path = os.path.join(path, player_name, "training")
        statistics_path = os.path.join(path, player_name, "statistics")
        
        # Ensure directories exist
        os.makedirs(training_path, exist_ok=True)
        os.makedirs(statistics_path, exist_ok=True)
        
        # Determine the next available file number for training data
        training_files = os.listdir(training_path)
        training_numbers = [int(f.split('_')[-1].split('.')[0]) for f in training_files if f.startswith("training_data_")]
        next_training_number = max(training_numbers, default=0) + 1
        training_file = os.path.join(training_path, f"training_data_{next_training_number}.json")
        
        # Log training data
        training_data = globals()[training_data_func](player_info, info, **training_data_func_args)
        with open(training_file, "w") as f:
            json.dump(training_data, f, indent=4)
        
        # Determine the next available file number for metrics
        metrics_files = os.listdir(statistics_path)
        metrics_numbers = [int(f.split('_')[-1].split('.')[0]) for f in metrics_files if f.startswith("metrics_")]
        next_metrics_number = max(metrics_numbers, default=0) + 1
        metrics_file = os.path.join(statistics_path, f"metrics_{next_metrics_number}.json")
        
        # Log metrics
        metrics = globals()[metrics_func](player_info, info, **metrics_func_args)
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
