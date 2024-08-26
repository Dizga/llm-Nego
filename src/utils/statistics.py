import json
import numpy as np
import hydra
from datetime import datetime
import os
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

# local imports
from utils.dond_logger import DondLogger
from environments.dond_game import DondGame
from environments.dond_player import DondPlayer
from agents.hf_agent import HfAgent
from agents.dummy_hf_agent import DummyHfAgent
from agents.oai_agent import OaiAgent

def set_itr_stats(self):
    """
    Computes statistics for the current iteration.

    Returns:
        dict: A dictionary containing statistics of the current iteration.
    """
    # self.iteration_stats = {
    #     "Iteration": self.iteration_nb,
    #     "Agreement Percentage": self.game_log['agreement_reached'].mean() * 100 if not self.game_log['agreement_reached'].empty else 0,
    #     "Score Variance player_0": self.game_log['player_0_score'].var() if not self.game_log['player_0_score'].empty else 0,
    #     "Score Variance player_1": self.game_log['player_1_score'].var() if not self.game_log['player_1_score'].empty else 0,
    #     "Mean Score player_0": self.game_log['player_0_score'].mean() if not self.game_log['player_0_score'].empty else 0,
    #     "Mean Score player_1": self.game_log['player_1_score'].mean() if not self.game_log['player_1_score'].empty else 0
    # }

def log_itr_stats(self):
    """
    Logs statistics for the current iteration and saves them to a CSV file.
    """
    
    # self.set_itr_stats()
    # iteration = self.iteration_stats['Iteration']

    # if iteration in self.statistics['Iteration'].values:
    #     self.statistics.loc[self.statistics['Iteration'] == iteration, :] = pd.DataFrame([self.iteration_stats])
    # else:
    #     self.statistics = pd.concat([self.statistics, pd.DataFrame([self.iteration_stats])], ignore_index=True)
    # self.statistics.to_csv(self.statistics_file, index=False)
