import os
import json
import logging
import logging.config
from datetime import datetime
import pandas as pd
import torch


class DondLogger:
    """
    Logger class for logging game data, metrics, and statistics.
    """
    def __init__(self, out_dir):
        """
        Initializes the Logger.

        Args:
            out_dir (str): The output directory where logs will be saved.
        """
        self.run_dir = out_dir
        self.datenow = datetime.now().strftime('%Y_%m_%d_%H_%M')

        self.iteration_nb = 0
        self.game_nb = 0
        self.round_nb = 0

    def new_iteration(self)-> str:
        """
        Starts a new iteration, resets metrics, and logs stats for the previous iteration.
        Returns:
            Path of folder where data is being logged.
        """
        self.iteration_nb += 1
        self.game_nb = 0
        self.it_folder = os.path.join(self.run_dir, f"iteration_{self.iteration_nb:02d}")
        os.makedirs(self.it_folder, exist_ok=True)
        # Reset metrics for the new iteration
        self.game_log = pd.DataFrame()
        self.game_log_file = os.path.join(self.it_folder, "games.csv")
        return self.it_folder

    def new_game(self):
        self.game_nb += 1
        self.round_nb = 0
        self.rounds_log = pd.DataFrame([])
        self.rounds_path = os.path.join(self.it_folder, 
                f"iter_{self.iteration_nb:02d}_game_{self.game_nb:04d}.csv")
        

    def log_game(self, summary, rounds, player_0_history, player_1_history):
        """
        Logs game data, saves player histories, and updates metrics.

        Args:
            game (dict): A dictionary containing game data.
        """
        
        player_0_game_name = f"player_0_iter_{self.iteration_nb:02d}_game_{self.game_nb:04d}.json"
        player_1_game_name = f"player_1_iter_{self.iteration_nb:02d}_game_{self.game_nb:04d}.json"

        os.makedirs(self.run_dir, exist_ok=True)

        with open(os.path.join(self.it_folder, player_0_game_name), 'w') as f:
            json.dump(player_0_history, f, indent=4)

        with open(os.path.join(self.it_folder, player_1_game_name), 'w') as f:
            json.dump(player_1_history, f, indent=4)

        summary['player_0_path'] = player_0_game_name
        summary['player_1_path'] = player_1_game_name
        summary['rounds_path'] = self.rounds_path

        # Log global game metrics
        self.game_log = pd.concat([self.game_log, pd.DataFrame([summary])], ignore_index=True)
        self.game_log.to_csv(self.game_log_file, index=False)

        # Log every round
        for round in rounds: self.log_round(round)


    def log_round(self, round: dict):
        """
        Logs game data, saves player histories, and updates metrics.

        Args:
            game (dict): A dictionary containing game data.
        """
        # Log round metrics
        self.rounds_log = pd.concat([self.rounds_log, pd.DataFrame([round])], ignore_index=True)
        self.rounds_log.to_csv(self.rounds_path, index=False)


    def setup_logging(self, config_file):
        """
        Sets up logging configuration from a config file.

        Args:
            config_file (str): Path to the logging configuration file.
        """
        logging.config.fileConfig(config_file, defaults={'date': self.datenow})

    def log_info(self, message: str):
        """
        Logs an info message.

        Args:
            message (str): The message to log.
        """
        logging.info(message)

    def log_debug(self, message: str):
        """
        Logs a debug message.

        Args:
            message (str): The message to log.
        """
        logging.debug(message)

    def log_warning(self, message: str):
        """
        Logs a warning message.

        Args:
            message (str): The message to log.
        """
        logging.warning(message)

    def log_error(self, message: str):
        """
        Logs an error message.

        Args:
            message (str): The message to log.
        """
        logging.error(message)

    def save_player_messages(self, player_name: str, messages: list):
        """
        Saves player messages to a JSON file.

        Args:
            player_name (str): The name of the player.
            messages (list): A list of messages from the player.
        """
        file_path = os.path.join(self.run_dir, f"{player_name}.json")
        with open(file_path, 'w') as f:
            json.dump(messages, f, indent=4)
