import os
import json
import logging
import logging.config
from datetime import datetime
from omegaconf import OmegaConf
import pandas as pd

class Logger:
    def __init__(self, name, cfg):
        self.datenow = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = os.path.join(cfg.log_dir, self.datenow) #TODO: add name
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = logging.getLogger(name)
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
        self.dataframe = pd.DataFrame(columns=columns)
        self.games_log_file = os.path.join(self.log_dir, 'games_log.txt')

    def log_game(self, game: dict, it: int, game_nb: int):
        p0_history = game.pop("p0_history")
        p1_history = game.pop("p1_history")

        p0_game_name = f"{game['player']}_GAME_{it}_{game_nb}_{self.datenow}.json"
        p1_game_name = f"{game['player']}_GAME_{it}_{game_nb}_{self.datenow}.json"

        os.makedirs(self.log_dir, exist_ok=True)

        with open(os.path.join(self.log_dir, p0_game_name), 'w') as f:
            json.dump(p0_history, f)

        with open(os.path.join(self.log_dir, p1_game_name), 'w') as f:
            json.dump(p1_history, f)

        game['p0_file'] = p0_game_name
        game['p1_file'] = p1_game_name

        self.dataframe = self.dataframe.append(game, ignore_index=True)
        self.dataframe.to_csv(self.dataframe_file)

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
        file_path = os.path.join(self.log_dir, f"{player_name}.json")
        with open(file_path, 'w') as f:
            json.dump(messages, f, indent=4)
