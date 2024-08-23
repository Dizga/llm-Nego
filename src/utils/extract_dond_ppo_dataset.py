import json
import numpy as np
import hydra
from datetime import datetime
import os
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import pandas as pd

# local imports
from utils.dond_logger import DondLogger
from environments.dond_game import DondGame
from environments.dond_instructor import DondInstructor
from agents.hf_agent import HfAgent
from agents.dummy_hf_agent import DummyHfAgent
from agents.oai_agent import OaiAgent



def extract_hf_ppo_dataset(folder_path: str, p0=True, full_context=True):
        """
        Args:
            file (str): Location of the csv / dataframe for the iteration
        """
        
        gm_messages_path_df_column = "rounds_path"

        if p0: 
            gm_messages_path_df_column = "p0_path"
            mg_rewards_df_column = "p0_return"
        else: 
            gm_messages_path_df_column = "p1_path"
            mg_rewards_df_column = "p1_return"

        # get jsons list
        queries = []
        responses = []
        scores = []

        # get all the games 
        games_info_df = pd.read_csv(os.path.join(folder_path, 'games.csv')) 
        games_info = games_info_df.to_dict(orient='records')

        # TODO: only analyse the games from the right player

        for game_info in games_info:

            # get game returns
            game_path = game_info['rounds_path']
            rounds_metrics_df = pd.read_csv(game_path)  # get rounds dataframe
            mg_rewards = rounds_metrics_df[mg_rewards_df_column].tolist()

            # get game conversation
            conv_path = os.path.join(folder_path, game_info[gm_messages_path_df_column])
            with open(conv_path, 'r') as file:
                game = json.load(file)

            context = []
            count = -1

            # extract queries, responses and scores
            for message in game:
                if message['role'] == "assistant":
                    queries.append(context)
                    responses.append(message)
                    scores.append(mg_rewards[count])
                elif message['is_new_round']:
                    count += 1
                context.append(message)

        return queries, responses, scores