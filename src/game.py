import json
from typing import List, Dict, Union
from utils import calculate_remaining_items, calculate_end_probability, calculate_rewards
from agents import GPTAgent

def generate_prompt(
    turn: int, expected_turns: int, state: Dict[str, Union[int, List[int], Dict[str, List[int]]]], 
    player_turn: str, opponent_turn: str, opponent_proposal: List[int] = None, 
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

    if opponent_proposal is None:
        opponent_proposal_text = "There is no opponent proposal yet. Make a proposal."
    else:
        remaining_items = calculate_remaining_items(item_quantities, opponent_proposal)
        remaining_items_text = ", ".join([f"{remaining_items[i]} of item_{i+1}" for i in range(len(remaining_items))])
        opponent_message = f"""{'Your opponent said: "' + opponent_message + '"'}""" if opponent_message else ""
        opponent_proposal_text = (
            f"The opponent's proposal is to take "
            + ", ".join([f"{opponent_proposal[i]} of item_{i+1}" for i in range(len(opponent_proposal))]) +
            f" and leave you with {remaining_items_text}. "
            f"{opponent_message}\n"
            "Reason about the current state of the game, then choose to accept or decline the proposal. "
            "If you decline this proposal, reason about a new proposal. "
            "If you decide that the best option is to accept your opponent's proposal, clearly say it at the end of your reasoning."
        )

    return (
        f"It is Turn {turn}. The probability of the game ending is {game_ending_prob:0.2f}. There are {item_info}. "
        f"Your utility values for the items are: {your_utilities}. "
        f"The opponent's utility values for the items are: {opponent_utilities}. "
        f"{opponent_proposal_text} "
    )

def generate_json_prompt(types_of_items: int) -> str:
    """Generate a JSON prompt for the player's proposal."""
    item_info = ", ".join(["int" for _ in range(types_of_items)])

    proposal_template = f"""{{
        "accept_opponent_proposal": true | false,
        "my_proposal": null | List[{item_info}],
        "comm_channel": null | str
    }}"""

    return (f"Return your answer as a valid JSON string following this template: {proposal_template}, 'my_proposal' should be a list of length {types_of_items}. "
            "Anything you want to say to your opponent, to share information or try to influence his decision for example, you can write in the 'comm_channel'. "
            "No explanation needed. No Markdown needed")

def nego_game(
    state: Dict[str, Union[int, List[int], Dict[str, List[int]]]], turns: int, expected_turns: int, 
    player1: GPTAgent, player2: GPTAgent, logger
) -> Dict[str, int]:
    """Run the negotiation game."""
    current_proposal = None
    current_communication = None
    max_retries = 3

    for turn in range(turns):
        logger.log_info(f"Turn {turn}")

        for player, opponent in [(player1, player2), (player2, player1)]:
            reasoning_prompt = generate_prompt(turn, expected_turns, state, player.name, opponent.name, current_proposal, current_communication)
            logger.log_debug(f"{player.name}: {reasoning_prompt}")
            player.add_message('user', reasoning_prompt)
            player()
            produce_json_prompt = generate_json_prompt(state['type_of_items'])
            player.add_message('user', produce_json_prompt)

            retries = 0
            while retries < max_retries:
                try:
                    player_response = player()
                    player_proposal = json.loads(player_response)
                    if player_proposal["my_proposal"] is not None and len(player_proposal["my_proposal"]) != state['type_of_items']:
                        retries += 1
                        logger.log_warning(f"Error in proposal from {player.name} response. Retry {retries} of {max_retries}.")
                        player.add_message("user", f"Invalid proposal. Your proposal should be a list of length {state['type_of_items']}. Please try again.")
                    else:
                        break
                except json.JSONDecodeError:
                    retries += 1
                    logger.log_warning(f"Error decoding JSON from {player.name} response. Retry {retries} of {max_retries}.")
                    player.add_message("user", "Invalid response. Please try again and return a valid JSON string.")

            if retries == max_retries:
                logger.log_error(f"Error decoding JSON from {player.name} after {max_retries} retries.")
                raise ValueError(f"Error decoding JSON from {player.name} after {max_retries} retries.")

            if player_proposal["accept_opponent_proposal"]:
                logger.log_info("Game ended with acceptance.")
                logger.log_info(f"Final proposal: {current_proposal}")
                return calculate_rewards(state, player.name, opponent.name, current_proposal)

            current_proposal = player_proposal["my_proposal"]
            current_communication = player_proposal["comm_channel"]

            logger.log_info(f"{player.name} proposal {current_proposal}.")

    return {player1.name: 0, player2.name: 0}
