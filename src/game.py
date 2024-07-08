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
    type_of_items, item_quantities, utilities, _ = state.values()

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
            f" and leave you with {remaining_items_text}.\n"
            f"{opponent_message}\n"
            "Reason about the current state of the game, then choose to accept or decline the proposal. "
            "If you decline this proposal, reason about a new proposal. "
            "If you decide that the best option is to accept your opponent's proposal, clearly say it at the end of your reasoning. "
            "Do not answer in JSON format yet. "
        )

    return (
        f"It is Turn {turn}. The probability of the game ending because the maximum number of turn is reached is {game_ending_prob:0.2f}.\n"
        f"There are {item_info}.\n"
        f"Your utility values for the items are: {your_utilities}.\n"
        f"The opponent's utility values for the items are: {opponent_utilities}.\n"
        f"{opponent_proposal_text} "
    )

def generate_json_prompt(types_of_items: int, comm_channel: bool) -> str:
    """Generate a JSON prompt for the player's proposal."""
    item_info = ", ".join(["int" for _ in range(types_of_items)])

    comm_prompt = ""

    if comm_channel:
        proposal_template = ("{{\n"
                            '\t"accept_opponent_proposal": true | false,\n'
                            f'\t"my_proposal": null | List[{item_info}],\n'
                            '\t"comm_channel": null | str\n'
                            '}}')
        comm_prompt = "Anything you want to say to your opponent, to share information or try to influence his decision for example, you can write in the 'comm_channel'. "
    else:
        proposal_template = ("\n{{\n"
                            '\t"accept_opponent_proposal": true | false,\n'
                            f'\t"my_proposal": null | List[{item_info}]\n'
                            '}}\n')

    return (f"Return your answer as a valid JSON string following this template:\n{proposal_template},\n'my_proposal' should be a list of length {types_of_items}. "
            f"{comm_prompt}" 
            "No explanation needed. No Markdown needed, your answer should start with '{'.")

def nego_game(
    state: Dict[str, Union[int, List[int], Dict[str, List[int]]]], expected_turns: int, 
    player1: GPTAgent, player2: GPTAgent, logger, comm_channel: bool = True
) -> Dict[str, int]:
    """Run the negotiation game."""

    turns = state['turns']
    proposals = []
    convo = []
    current_proposal = None
    current_communication = None
    max_retries = 3

    for turn in range(turns):
        logger.log_info(f"Turn {turn}")

        for player, opponent in [(player1, player2), (player2, player1)]:
            reasoning_prompt = generate_prompt(turn, expected_turns, state, player.name, opponent.name, current_proposal, current_communication)
            logger.log_debug(f"{player.name}: {reasoning_prompt}")
            player.add_message('user', reasoning_prompt)
            response = player()
            convo.append({"content": response, "name": player.name, "type": "reasoning"})
            produce_json_prompt = generate_json_prompt(state['type_of_items'], comm_channel)
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
                        convo.append({"content": player_proposal, "name": player.name, "type": "proposal"})
                        break
                except json.JSONDecodeError:
                    retries += 1
                    logger.log_warning(f"Error decoding JSON from {player.name} response. Retry {retries} of {max_retries}.")
                    player.add_message("user", "Invalid response. Please try again and return a valid JSON string. No explanation needed. No Markdown needed, your answer should start with '{'.")

            if retries == max_retries:
                logger.log_error(f"Error decoding JSON from {player.name} after {max_retries} retries.")
                raise ValueError(f"Error decoding JSON from {player.name} after {max_retries} retries.")

            if player_proposal["accept_opponent_proposal"]:
                logger.log_info("Game ended with acceptance.")
                logger.log_info(f"Final proposal: {current_proposal}")
                return calculate_rewards(state, player.name, opponent.name, current_proposal), convo

            current_proposal = player_proposal["my_proposal"]
            current_communication = player_proposal["comm_channel"] if comm_channel else None
            proposals.append(player_proposal)

            logger.log_info(f"{player.name} proposal {current_proposal}.")

    return {player1.name: 0, player2.name: 0}, convo
