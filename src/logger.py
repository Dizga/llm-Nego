import os
import json
import logging
import logging.config
from datetime import datetime
from omegaconf import OmegaConf
import pandas as pd

class Logger:
    def __init__(self):
        self.datenow = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = f"DATA/RUN_{self.datenow}" #TODO: add name
        os.makedirs(self.run_dir, exist_ok=True)
        #self.logger = logging.getLogger(name)
        columns = [
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
        self.metrics = pd.DataFrame(columns=columns)
        columns = [
            "Iteration",
            "Agreement Percentage",
            "Score Mean",
            "Score Variance",
            "Score Mean P0",
            "Score Mean P1"
        ]
        self.statistics = pd.DataFrame(columns)
        self.iteration = 0

    def new_iteration(self):
        self.iteration += 1
        self.game_nb = 1
        self.it_folder = os.path.join(self.run_dir, f"ITERATION_{self.iteration}")
        os.makedirs(self.run_dir, exist_ok=True)
        # Reset metrics
        self.metrics = self.metrics[:]
        self.metrics_file = os.path.join(self.it_folder, "metrics.csv")
        # compute stats of past iteration
        self.log_stats()

    def log_stats(self):
        # TODO
        pass


    def log_game(self, game: dict):
        p0_history = game.pop("p0_history")
        p1_history = game.pop("p1_history")

        p0_game_name = f"{game['player']}_GAME_{self.iteration}_{self.game_nb}_{self.datenow}.json"
        p1_game_name = f"{game['player']}_GAME_{self.iteration}_{self.game_nb}_{self.datenow}.json"

        os.makedirs(self.run_dir, exist_ok=True)

        with open(os.path.join(self.run_dir, p0_game_name), 'w') as f:
            json.dump(p0_history, f)

        with open(os.path.join(self.run_dir, p1_game_name), 'w') as f:
            json.dump(p1_history, f)

        game['p0_file'] = p0_game_name
        game['p1_file'] = p1_game_name

        self.metrics = self.metrics.append(game, ignore_index=True)
        self.metrics.to_csv(self.metrics_file)
        self.game_nb +=1

    def comp_stats_and_log(self):
        pass

    def setup_logging(self, config_file):
        logging.config.fileConfig(config_file, defaults={'date': self.datenow})

    def log_info(self, message: str):
        self.logger.info(message)

    def log_debug(self, message: str):
        self.logger.debug(message)

    def log_warning(self, message: str):
        self.logger.warning(message)

    def log_error(self, message: str):
        self.logger.error(message)

    def save_player_messages(self, player_name: str, messages: list):
        file_path = os.path.join(self.run_dir, f"{player_name}.json")
        with open(file_path, 'w') as f:
            json.dump(messages, f, indent=4)
