import json
import numpy as np
import hydra
from datetime import datetime
import os
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

# local imports
from environments.dond_game import DondGame
from environments.dond_player import DondPlayer
from models.hf_agent import HfAgent
from models.dummy_hf_agent import DummyHfAgent
from models.oai_agent import OaiAgent


class BcDondTrainer:
    def __init__(self, 
                 iterations_per_run, 
                 games_per_iteration, 
                 game: DondGame, 
                 train_type: str,
                 player_0: DondPlayer, 
                 player_1: DondPlayer
                 ):

        self.iterations_per_run = iterations_per_run
        self.games_per_iteration = games_per_iteration
        self.game = game
        self.player_0 = player_0
        self.player_1 = player_1
        self.train_type = train_type

    def run_iterations(self):
        for _ in range(self.iterations_per_run):
            folder_path = self.logger.new_iteration()
            for _ in range(self.games_per_iteration):
                self.run_game()
            if self.train_type == "ppo":
                self.train_agents_ppo(folder_path)
            else:
                self.train_agents_bc(folder_path)


    def run_game(self):
        self.logger.log_info("Game started.")
        self.logger.reset_game()
        players = [self.player_0, self.player_1]
        self.player_0.reset_game()
        self.player_1.reset_game()
        game_state = self.game.reset()
        player_id = 0
        while not game_state['game_ended']:
            if game_state['reset_round']:
                self.player_0.reset_round()
                self.player_1.reset_round()
            is_finalization, content = players[player_id].play_move(game_state)
            game_state = self.game.step(content, is_finalization=is_finalization)
            player_id = (player_id + 1) % 2
            
        # while True:
        #     if self.player_0.play_move(): break
        #     if self.player_1.play_move(): break
        self.logger.export_match(*self.game.export(), 
                             self.player_0.get_history(), 
                             self.player_1.get_history())
        self.logger.log_info("Game completed.")

    def train_agents_ppo(self, folder_path):

        self.logger.log_info("PPO training started.")
        
        # Train player 0
        self.player_0.agent.init_ppo_trainer()
        queries, responses, scores = self.logger.extract_hf_ppo_dataset(folder_path, player_0=True)
        self.player_0.agent.train_ppo_json(queries, responses, scores)

        # Train player 1
        self.player_1.agent.init_ppo_trainer()
        queries, responses, scores = self.logger.extract_hf_ppo_dataset(folder_path, player_0=False)
        self.player_1.agent.train_ppo_json(queries, responses, scores)

        self.logger.log_info("PPO training ended.")


    def train_agents_bc(self):
        """Train the agents on the last iteration."""
        # TODO: update this function
        metrics = self.logger.metrics # Extract dataframe with data for each game
        mean_score_player_0 = self.logger.iteration_stats['Mean Score player_0'] # Get the mean score of the current iteration
        mean_score_player_1 = self.logger.iteration_stats['Mean Score player_1'] # Get the mean score of the current iteration
        # Filter games with score better than the mean score
        filtered_player_0 = metrics[metrics['player_0_reward'] >= mean_score_player_0]
        filtered_player_1 = metrics[metrics['player_1_reward'] >= mean_score_player_1]
        player_0_filtered_files = [self.logger.it_folder + '/' + element for element in filtered_player_0['player_0_file'].tolist()]
        player_1_filtered_files = [self.logger.it_folder + '/' + element for element in filtered_player_1['player_1_file'].tolist()]
        player_0_filtered_jsons = [json.load(open(file_path, 'r')) for file_path in player_0_filtered_files]
        player_1_filtered_jsons = [json.load(open(file_path, 'r')) for file_path in player_1_filtered_files]
        self.player_0.agent.train(player_0_filtered_jsons)
        self.player_1.agent.train(player_1_filtered_jsons)




def run_bc_dond(cfg): # TODO: change name
    
    # Make output directory
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_directory = hydra_cfg['runtime']['output_dir']
    os.makedirs(output_directory, exist_ok=True)

    # Get input and output handler for DonD

    # Get instance which handles the game
    game = DondGame(**cfg.game)

    # Get the player/Player 0
    if cfg.players.player_0.type == "hf": agent_0 = HfAgent(**cfg.players.player_0.agent_args)
    if cfg.players.player_0.type == "dummy_hf": agent_0 = DummyHfAgent(**cfg.players.player_0.agent_args)
    elif cfg.players.player_0.type == "oai": agent_0 = HfAgent(**cfg.players.player_0.agent_args)
    player_0 = DondPlayer(
        **cfg.players.player_0.player_args, dond_game=game,
        agent=agent_0, player_type="player_0"
    )

    # Get player/Player 1
    if cfg.players.shared_model: cfg.players.player_1.agent_args.model_name = None
    if cfg.players.player_1.type == "hf": agent_1 = HfAgent(**cfg.players.player_1.agent_args)
    elif cfg.players.player_1.type == "dummy_hf": agent_1 = DummyHfAgent(**cfg.players.player_1.agent_args)
    elif cfg.players.player_1.type == "oai": agent_1 = HfAgent(**cfg.players.player_1.agent_args)
    if cfg.players.shared_model: agent_1.model = agent_0.model

    player_1 = DondPlayer(
        **cfg.players.player_0.player_args, dond_game=game,
        agent=agent_1, player_type="player_1"
    )

    # Start training
    trainer = BcDondTrainer(
        **cfg.training, game=game,
        player_0=player_0,
        player_1=player_1,
        logger=logger
    )
    trainer.run_iterations()


@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    if os.path.exists('conf/local.yaml'):
        local_cfg = OmegaConf.load('conf/local.yaml')
        cfg = OmegaConf.merge(cfg, local_cfg)
    run_bc_dond(cfg)
if __name__ == "__main__": main()
