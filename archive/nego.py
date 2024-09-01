from datetime import datetime
import os
import random
import json
import logging
import logging.config
import numpy as np
from scipy.stats import poisson
from typing import List, Dict, Union
from agents import HFAgent, GPTAgent


datenow = datetime.now().strftime('%Y%m%d_%H%M%S')
os.mkdir(f'logs/{datenow}') 

logging.config.fileConfig('logging.conf', defaults={'date':datenow})
logger = logging.getLogger(__name__)

def generate_initial_state(
    items_min: int = 3, items_max: int = 3, quantity_min: int = 1, 
    quantity_max: int = 5, utility_min: int = 1, utility_max: int = 5, 
    player_1_name: str = 'player_1', player_2_name: str = 'player_2'
) -> Dict[str, Union[int, List[int], Dict[str, List[int]]]]:
    """Generate the initial state of the negotiation game."""
    type_of_items = random.randint(items_min, items_max)
    item_quantities = [random.randint(quantity_min, quantity_max) for _ in range(type_of_items)]
    player_1_utility_values = [random.randint(utility_min, utility_max) for _ in range(type_of_items)]
    player_2_utility_values = [random.randint(utility_min, utility_max) for _ in range(type_of_items)]
    return {
        "type_of_items": type_of_items,
        "item_quantities": item_quantities,
        "utilities": {
            player_1_name: player_1_utility_values,
            player_2_name: player_2_utility_values
        }
    }

def calculate_remaining_items(item_quantities: List[int], finalization: List[int]) -> List[int]:
    """Calculate the remaining items after a finalization."""
    return [item_quantities[i] - finalization[i] for i in range(len(item_quantities))]

def generate_prompt(
    turn: int, expected_turns: int, state: Dict[str, Union[int, List[int], Dict[str, List[int]]]], 
    player_turn: str, opponent_turn: str, opponent_finalization: List[int] = None, 
    opponent_message: str = None
) -> str:
    """Generate a prompt for the player's turn."""
    type_of_items, item_quantities, utilities = state.values()

    player_utilities = utilities[player_turn]
    opponent_utilities = utilities[opponent_turn]

    item_info = ", ".join([f"{item_quantities[i]} units of item_{i+1}" for i in range(len(item_quantities))])
    your_utilities = ", ".join([f"{player_utilities[i]} for item_{i+1}" for i in range(len(player_utilities))])
    opponent_utilities = ", ".join([f"{opponent_utilities[i]} for item_{i+1}" for i in range(len(opponent_utilities))])

    game_ending_prob = calculate_end_probability(turn, expected_turns)

    if opponent_finalization is None:
        opponent_finalization_text = "There is no opponent finalization yet. Make a finalization."
    else:
        remaining_items = calculate_remaining_items(item_quantities, opponent_finalization)
        remaining_items_text = ", ".join([f"{remaining_items[i]} of item_{i+1}" for i in range(len(remaining_items))])
        opponent_message = f"""{'Your opponent said: "' + opponent_message + '"'}""" if opponent_message else ""
        opponent_finalization_text = (
            f"The opponent's finalization is to take "
            + ", ".join([f"{opponent_finalization[i]} of item_{i+1}" for i in range(len(opponent_finalization))]) +
            f" and leave you with {remaining_items_text}. "
            f"{opponent_message}\n"
            "Reason about the current state of the game, then choose to accept or decline the finalization. "
            "If you decline this finalization, reason about a new finalization. "
            "If you decide that the best option is to accept your opponent's finalization, clearly say it at the end of your reasoning."
        )

    return (
        f"It is Turn {turn}. The probability of the game ending is {game_ending_prob:0.2f}. There are {item_info}. "
        f"Your utility values for the items are: {your_utilities}. "
        f"The opponent's utility values for the items are: {opponent_utilities}. "
        f"{opponent_finalization_text} "
    )

def calculate_end_probability(turn: int, lambda_: float) -> float:
    """Calculate the probability that the game ends on the given turn."""
    P_T_eq_t = poisson.pmf(turn, lambda_)
    P_T_ge_t = 1 - poisson.cdf(turn - 1, lambda_)
    
    if P_T_ge_t == 0:
        return 1.0
    
    P_end_at_t_given_reached_t = P_T_eq_t / P_T_ge_t
    return P_end_at_t_given_reached_t

instruction_prompt = (
    "You are a player in a two-player negotiation game. "
    "Your goal is to maximize your own reward by proposing how to divide a set of items. "
    "Your reward is equal to the number of items you receive multiplied by the utility for each. "
    "You and your opponent will take turns giving finalizations of how to divide the items. "
    "Each finalization specifies how many of each item it wants, leaving the remaining items for the other player.\n"
    "Before giving a finalization, each player can choose to accept the opponent's last finalization and end the game, "
    "the items would then be divided according to the accepted finalization. "
    "If no finalization is accepted after a random amount of turns, the game ends with both players receiving a reward of 0.\n"
)

def generate_json_prompt(types_of_items: int) -> str:
    """Generate a JSON prompt for the player's finalization."""
    item_info = ", ".join(["int" for _ in range(types_of_items)])

    finalization_template = f"""{{
        "accept_opponent_finalization": true | false,
        "my_finalization": null | List[{item_info}],
        "comm_channel": null | str
    }}"""

    return (f"Return your answer as a valid JSON string following this template: {finalization_template}, 'my_finalization' should be a list of length {types_of_items}. "
            "Anything you want to say to your opponent, to share information or try to influence his decision for example, you can write in the 'comm_channel'. "
            "No explanation needed. No Markdown needed")

def calculate_rewards(
    state: Dict[str, Union[int, List[int], Dict[str, List[int]]]], player: str, opponent: str, finalization: List[int]
) -> Dict[str, int]:
    """Calculate the rewards for both players based on the finalization."""
    type_of_items, item_quantities, utilities = state.values()

    player_utilities = utilities[player]
    opponent_utilities = utilities[opponent]

    remaining_items = calculate_remaining_items(item_quantities, finalization)

    return {
        player: sum([remaining_items[i] * player_utilities[i] for i in range(type_of_items)]),
        opponent: sum([finalization[i] * opponent_utilities[i] for i in range(type_of_items)])
    }

def nego_game(
    state: Dict[str, Union[int, List[int], Dict[str, List[int]]]], turns: int, expected_turns: int, 
    player1: HFAgent, player2: HFAgent
) -> Dict[str, int]:
    """Run the negotiation game."""
    current_finalization = None
    current_communication = None
    max_retries = 3

    for turn in range(turns):
        logger.info(f"Turn {turn}")

        for player, opponent in [(player1, player2), (player2, player1)]:
            reasoning_prompt = generate_prompt(turn, expected_turns, state, player.name, opponent.name, current_finalization, current_communication)
            logger.debug(f"{player.name}: {reasoning_prompt}")
            player.add_message('user', reasoning_prompt)
            player()
            produce_json_prompt = generate_json_prompt(state['type_of_items'])
            player.add_message('user', produce_json_prompt)

            retries = 0
            while retries < max_retries:
                try:
                    player_response = player()
                    player_finalization = json.loads(player_response)
                    if player_finalization["my_finalization"] is not None and len(player_finalization["my_finalization"]) != state['type_of_items']:
                        retries += 1
                        logger.warning(f"Error in finalization from {player.name} response. Retry {retries} of {max_retries}.")
                        player.add_message("user", f"Invalid finalization. Your finalization should be a list of length {state['type_of_items']}. Please try again.")
                    else:
                        break
                except json.JSONDecodeError:
                    retries += 1
                    logger.warning(f"Error decoding JSON from {player.name} response. Retry {retries} of {max_retries}.")
                    player.add_message("user", "Invalid response. Please try again and return a valid JSON string.")

            if retries == max_retries:
                logger.error(f"Error decoding JSON from {player.name} after {max_retries} retries.")
                raise ValueError(f"Error decoding JSON from {player.name} after {max_retries} retries.")

            if player_finalization["accept_opponent_finalization"]:
                logger.info("Game ended with acceptance.")
                logger.info(f"Final finalization: {current_finalization}")
                return calculate_rewards(state, player.name, opponent.name, current_finalization)

            current_finalization = player_finalization["my_finalization"]
            current_communication = player_finalization["comm_channel"]

            logger.info(f"{player.name} finalization {current_finalization}.")

    return {player1.name: 0, player2.name: 0}

# player1 = HFAgent("player_1")
player1 = GPTAgent("player_1")
player2 = GPTAgent("player_2")

player1.add_system_message(instruction_prompt)
player2.add_system_message(instruction_prompt)

state = generate_initial_state()
logger.info("State:")
logger.info(state)

expected_turns = 5
num_turns = np.random.poisson(expected_turns)
logger.info(f"There will be {num_turns} turns.")

rewards = nego_game(state, num_turns, expected_turns, player1, player2)

with open(f"logs/{datenow}/player_1.json", "w") as f:
    json.dump(player1.messages, f, indent=4)
with open(f"logs/{datenow}/player_2.json", "w") as f:
    json.dump(player2.messages, f, indent=4)

logger.info("Rewards:")
logger.info(rewards)
