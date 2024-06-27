import random
from typing import List, Dict, Union
import numpy as np
from scipy.stats import poisson

def generate_initial_state(
    items_min: int = 3, items_max: int = 3, quantity_min: int = 1, 
    quantity_max: int = 5, utility_min: int = 1, utility_max: int = 5, expected_turns = 5,
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
        },
        "turns": np.random.poisson(expected_turns)
    }

def calculate_remaining_items(item_quantities: List[int], proposal: List[int]) -> List[int]:
    """Calculate the remaining items after a proposal."""
    return [item_quantities[i] - proposal[i] for i in range(len(item_quantities))]

def calculate_end_probability(turn: int, lambda_: float) -> float:
    """Calculate the probability that the game ends on the given turn."""
    P_T_eq_t = poisson.pmf(turn + 1, lambda_)
    P_T_ge_t = 1 - poisson.cdf(turn, lambda_)
    
    if P_T_ge_t == 0:
        return 1.0
    
    P_end_at_t_given_reached_t = P_T_eq_t / P_T_ge_t
    return P_end_at_t_given_reached_t

def calculate_rewards(
    state: Dict[str, Union[int, List[int], Dict[str, List[int]]]], player: str, opponent: str, proposal: List[int]
) -> Dict[str, int]:
    """Calculate the rewards for both players based on the proposal."""
    type_of_items, item_quantities, utilities, _ = state.values()

    player_utilities = utilities[player]
    opponent_utilities = utilities[opponent]

    remaining_items = calculate_remaining_items(item_quantities, proposal)

    return {
        player: sum([remaining_items[i] * player_utilities[i] for i in range(type_of_items)]),
        opponent: sum([proposal[i] * opponent_utilities[i] for i in range(type_of_items)])
    }
