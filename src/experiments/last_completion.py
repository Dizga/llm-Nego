import hydra
import os
import logging
import time
from omegaconf import OmegaConf
import random
# local imports
from src.environments.dond_run_matches import run_matches
from environments.dond_game import DondGame
from models.hf_agent import HfAgent
from models.dummy_hf_agent import DummyHfAgent
from models.oai_agent import OaiAgent
from statistics import mean
from utils.plot_curves import plot_curves

from environments.dond_player import DondPlayerHandler
from training.extract_ppo_dataset import extract_ppo_dataset
from training.extract_sft_dataset import extract_sft_dataset
import copy


def get_data(game, player, agent, n_samples):
    state = game.get_state()
    assert state['has_finalized']
    player.set_usr_message(state)
    context = player.get_context()

    contexts = [copy.deepcopy(context) for _ in range(n_samples)]
    responses = agent.prompt(contexts)

    scores = []
    for i in range(n_samples):
        send_to_game, is_finalization, processed_response = player.process_model_response(responses[i], state)
        if not send_to_game: 
            scores.append(0)
            continue
        else:
            game_copy = copy.deepcopy(game)
            game_copy.step(processed_response, is_finalization)
            state_ = game_copy.get_state()
            if state_['agreement_reached_history'][-1]: scores.append(10)
            else: scores.append(0)
    assert len(scores) == n_samples
    responses = [[{'role':'assistant', 'content':r}] for r in responses]
    return contexts, responses, scores


def run_partial_game(game, player_0, player_1, agent):
    current_player = player_0
    other_player = player_1
    for _ in range(1, 1000):
        # Play one turn at a time
        state = game.get_state()
        send_to_game = False

        while not send_to_game: 
            current_player.set_usr_message(state)
            context = current_player.get_context()
            # Generate response using the agent
            response = agent.prompt([context])[0]
            send_to_game, is_finalization, processed_response = current_player.process_model_response(response, state)
        
        game.step(processed_response, is_finalization)

        if is_finalization:
            return game, other_player  # Game ends, return the game state and the other player

        # Swap players for the next turn
        current_player, other_player = other_player, current_player


def last_completion(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True, structured_config_mode='dict')
    NB_TRAINING_STEPS = cfg['N_TRAINING_STEPS']
    NB_SAMPLES = cfg['NB_SAMPLES']

    agent = HfAgent(**cfg['models']['llama']['init_args'])
    player_0 = DondPlayerHandler(player_name="player_a", **cfg['players']['player_a']['dond_player_args'])
    player_0.game_id = 0
    player_1 = DondPlayerHandler(player_name="player_b", **cfg['players']['player_b']['dond_player_args'])
    player_1.game_id = 1

    dond_game = DondGame(**cfg['dond_game_args'])
    dond_game, player = run_partial_game(dond_game, player_0, player_1, agent)

    mean_scores = []

    for _ in range(NB_TRAINING_STEPS):
        queries, responses, scores = get_data(dond_game, copy.deepcopy(player), agent, NB_SAMPLES)
        mean_score = mean(scores)
        mean_scores.append(mean_score)
        #assert mean_score > 0.0
        plot_curves(y_list=[mean_scores], plot_name='MEAN SCORE OVER PPO STEPS')
        agent.train_ppo(queries, responses, scores)

    logging.info('Experiment completed.')





        



