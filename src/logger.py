import os
import json
import logging
import logging.config
from datetime import datetime
import pandas as pd

class Logger:
    def __init__(self, out_dir):
        self.run_dir = out_dir
        self.datenow = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        columns_metrics = [
            'p0_score',
            'p1_score',
            'quantities',
            'p0_values',
            'p1_values',
            'p0_proposal',
            'p1_proposal',
            'reach_agreement',
            'p0_file',
            'p1_file'
        ]
        self.metrics = pd.DataFrame(columns=columns_metrics)
        
        columns_statistics = [
            "Iteration",
            "Agreement Percentage",
            "Score Variance P0",
            "Score Variance P1",
            "Mean Score P0",
            "Mean Score P1"
        ]
        self.statistics = pd.DataFrame(columns=columns_statistics)
        self.statistics_file = os.path.join(self.run_dir, "STATISTICS.csv")
        self.iteration = 0

    def new_iteration(self):
        self.iteration += 1
        self.game_nb = 1
        self.it_folder = os.path.join(self.run_dir, f"ITERATION_{self.iteration}")
        os.makedirs(self.it_folder, exist_ok=True)
        # Reset metrics for the new iteration
        self.metrics = pd.DataFrame(columns=self.metrics.columns)
        self.metrics_file = os.path.join(self.it_folder, "metrics.csv")
        # Compute and log stats of the past iteration
        self.log_itr_stats()

    def get_itr_stats(self):
        # Logs stats for the current iteration
        self.iteration_stats = {
            "Iteration": self.iteration,
            "Agreement Percentage": self.metrics['reach_agreement'].mean() * 100 if not self.metrics['reach_agreement'].empty else 0,
            "Score Variance P0": self.metrics['p0_score'].var() if not self.metrics['p0_score'].empty else 0,
            "Score Variance P1": self.metrics['p1_score'].var() if not self.metrics['p1_score'].empty else 0,
            "Mean Score P0": self.metrics['p0_score'].mean() if not self.metrics['p0_score'].empty else 0,
            "Mean Score P1": self.metrics['p1_score'].mean() if not self.metrics['p1_score'].empty else 0
        }
        return self.iteration_stats

    def log_itr_stats(self):
        stats = self.get_itr_stats()
        self.statistics = pd.concat([self.statistics, pd.DataFrame([stats])], ignore_index=True)
        self.statistics.to_csv(self.statistics_file, index=False)

    def log_game(self, game: dict):
        p0_history = game.pop("p0_history")
        p1_history = game.pop("p1_history")

        p0_game_name = f"P0_GAME_{self.iteration}_{self.game_nb}_{self.datenow}_p0.json"
        p1_game_name = f"P1_GAME_{self.iteration}_{self.game_nb}_{self.datenow}_p1.json"

        os.makedirs(self.run_dir, exist_ok=True)

        with open(os.path.join(self.it_folder, p0_game_name), 'w') as f:
            json.dump(p0_history, f, indent=4)

        with open(os.path.join(self.it_folder, p1_game_name), 'w') as f:
            json.dump(p1_history, f, indent=4)

        game['p0_file'] = p0_game_name
        game['p1_file'] = p1_game_name

        self.metrics = pd.concat([self.metrics, pd.DataFrame([game])], ignore_index=True)
        self.metrics.to_csv(self.metrics_file, index=False)
        self.game_nb += 1

    def setup_logging(self, config_file):
        logging.config.fileConfig(config_file, defaults={'date': self.datenow})

    def log_info(self, message: str):
        logging.info(message)

    def log_debug(self, message: str):
        logging.debug(message)

    def log_warning(self, message: str):
        logging.warning(message)

    def log_error(self, message: str):
        logging.error(message)

    def save_player_messages(self, player_name: str, messages: list):
        file_path = os.path.join(self.run_dir, f"{player_name}.json")
        with open(file_path, 'w') as f:
            json.dump(messages, f, indent=4)
