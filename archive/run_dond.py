import json
import numpy as np
from logger import Logger
from DoND import DoND
from agents import NegoAgent
import hydra
from datetime import datetime
import os
from Player import DoNDPlayer
from hydra.core.hydra_config import HydraConfig

class TwoPlayerNegotiationTrainer:
    def __init__(self, iterations_per_run, games_per_iteration, game, player_0, player_1, logger):
        self.iterations_per_run = iterations_per_run
        self.games_per_iteration = games_per_iteration
        self.game = game
        self.player_0 = player_0
        self.player_1 = player_1
        self.logger = logger

    def train_agents(self):
        pass

    def run_iterations(self):
        for _ in range(self.iterations_per_run):
            self.logger.new_iteration()
            for _ in range(self.games_per_iteration):
                self.run_game()
            #self.train_agents()

    def run_game(self):
        self.logger.log_info("Game started.")
        self.logger.new_game()
        self.player_0.new_game()
        self.player_1.new_game()
        self.game.reset()
        while True:
            if not self.player_0.play_move(): break
            if not self.player_1.play_move(): break
        self.logger.log_game(self.game.export(), self.player_0.get_history(), self.player_1.get_history())
        self.logger.log_info("Game completed.")

        

class DoNDTrainer(TwoPlayerNegotiationTrainer):
    def train_agents(self):
        """Train the agents on the last iteration."""
        metrics = self.logger.metrics # Extract dataframe with data for each game
        mean_score_player_0 = self.logger.iteration_stats['Mean Score player_0'] # Get the mean score of the current iteration
        mean_score_player_1 = self.logger.iteration_stats['Mean Score player_1'] # Get the mean score of the current iteration
        # Filter games with score better than the mean score
        filtered_player_0 = metrics[metrics['player_0_score'] >= mean_score_player_0]
        filtered_player_1 = metrics[metrics['player_1_score'] >= mean_score_player_1]
        player_0_filtered_files = [self.logger.it_folder + '/' + element for element in filtered_player_0['player_0_file'].tolist()]
        player_1_filtered_files = [self.logger.it_folder + '/' + element for element in filtered_player_1['player_1_file'].tolist()]
        player_0_filtered_jsons = [json.load(open(file_path, 'r')) for file_path in player_0_filtered_files]
        player_1_filtered_jsons = [json.load(open(file_path, 'r')) for file_path in player_1_filtered_files]
        self.player_0.agent.train(player_0_filtered_jsons)
        self.player_1.agent.train(player_1_filtered_jsons)

@hydra.main(config_path="../conf", config_name="config")
def run_dond(cfg):

    # Make output directory
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_directory = hydra_cfg['runtime']['output_dir']
    os.makedirs(output_directory, exist_ok=True)

    logger = Logger(output_directory)
    game = DoND(max_turns=cfg.game.max_turns)

    agent_0 = NegoAgent(
        name="agent_0",
        device=cfg.device,
        model=cfg.player_0.model,
        tokenizer=cfg.player_0.tokenizer,
    )
    player_0 = DoNDPlayer(
        game_intro_file=cfg.player_0.game_intro_file,
        chain_of_thought_file=cfg.player_0.chain_of_thought,
        proposal_file=cfg.player_0.proposal_file,
        dond_game=game,
        agent=agent_0,
        player_type="player_0"
    )

    agent_1 = NegoAgent(
        name="agent_1",
        device=cfg.device,
        model=cfg.player_1.model,
        tokenizer=cfg.player_1.tokenizer,
    )
    player_1 = DoNDPlayer(
        game_intro_file=cfg.player_1.game_intro_file,
        chain_of_thought_file=cfg.player_1.chain_of_thought,
        proposal_file=cfg.player_0.proposal_file,
        dond_game=game,
        agent=agent_1,
        player_type="player_1"
    )

    trainer = DoNDTrainer(
        iterations_per_run=cfg.run.nb_iterations,
        games_per_iteration=cfg.run.games_per_iteration,
        game=game,
        player_0=player_0,
        player_1=player_1,
        logger=logger
    )

    trainer.run_iterations()

if __name__ == "__main__":
    run_dond()
