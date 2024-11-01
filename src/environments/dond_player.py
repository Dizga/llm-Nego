import json
import regex as re
import copy
# local imports
from environments.dond_game import DondGame
import math
from statistics import mean
import numpy as np

class DondPlayer:
    def __init__(
        self,
        player_name,
        allow_reasoning,
        game_intro_file,
        in_between_file,
        new_round_file,
        max_retries,
        finalization_file,
        mod_adpt_id,
    ):
        """
        Initializes the DoNDPlayer.

        Args:
            player_name (str): The name of the player.
            game_intro_file (str): Path to the file containing game introduction.
            in_between_file (str): Path to the file containing chain of thought instructions.
            new_round_file (str): Path to the file containing new round instructions.
            max_retries (int): Maximum number of retries allowed.
            finalization_file (str): Path to the file containing finalization instructions.
            mod_adpt_id (str): The name of the model used.
        """
        self.allow_reasoning = allow_reasoning
        self.player_name = player_name
        self.max_retries = max_retries
        self.mod_adpt_id = mod_adpt_id
        self.game_id = None  # ID of player in game
        self.new_game()

    def get_context(self):
        return self.context

    def get_augmented_context(self):
        return self.augmented_context

    def add_to_context(self, element: dict):
        self.context.append(element)
        self.augmented_context.append(element)

    def process_model_response(self, response, state):
        """
        Processes the response from the model and updates the game state.

        Args:
            response (str): The response from the model.
            state (dict): The current state of the game.

        Returns:
            tuple: A tuple containing:
                - send_to_game (bool): Indicates if the response should be sent to the game.
                - is_finalization (bool): Indicates if the response is a finalization.
                - processed_response (str or dict): The processed response.
        """
        # Initiate what will be returned
        processed_response = None
        send_to_game = False
        is_finalization = False

        # Process response. Check for errors.
        is_error, error_message, is_finalization, processed_response = self.process_response(
            response, state
        )

        if is_error:
            self.retries += 1
            self.error_message = error_message
            # Too many mistakes were made
            if self.retries > self.max_retries:
                self.error_message = False
                processed_response = "-------"
                send_to_game = True
                self.retries = 0

        else:
            self.retries = 0
            send_to_game = True

        # Add raw response to context
        model_response = {
            "role": "assistant",
            "content": response,
            "is_error": is_error,
            "is_finalization": is_finalization,
            "round_nb": state["round_number"],
        }

        self.add_to_context(model_response)

        return send_to_game, (is_finalization, processed_response)


    def set_usr_message(self, state):
        """
        Constructs a user message based on the current game state.

        Args:
            state (dict): The current state of the game.

        Returns:
            str: The constructed user message.
        """

        user_message = ""

        if self.error_message:
            user_message = self.error_message
            usr_prompt = {
                "role": "user",
                "content": user_message,
                "is_error": True,
                "round_nb": state["round_number"],
            }
            self.add_to_context(usr_prompt)
            self.error_message = None
            return

        if state["is_new_round"]:
            self.new_round()
            self.set_preplay_round_info(state)

        if state["is_new_game"]:
            user_message += create_game_intro_prompt(state)

        if state["is_new_round"] and not state["is_new_game"]:
            user_message += create_new_round_prompt(state)

        user_message += create_play_prompt(state)

        user_message = {
            "role": "user",
            "content": user_message,
            "is_error": False,
            "round_nb": state["round_number"],
        }
        self.add_to_context(user_message)

    def process_response(self, response, state):
        """
        Validates and extracts content from the response of the LLM player.

        Args:
            response (str): The response from the LLM player.
            state (dict): The current state of the game.

        Returns:
            tuple: A tuple containing:
                - is_error (bool): Indicates if there is an error.
                - error_message (str): The error message if there is an error, otherwise an empty string.
                - is_finalization (bool): Indicates if the response is a finalization.
                - processed_response (str or dict): The extracted message or finalization details.
        """
        errors = []

        # Check for presence of <reason> tag if allow_reasoning is not None
        if self.allow_reasoning is not False:
            if "<reason>" not in response or "</reason>" not in response:
                errors.append("Missing <reason>...</reason> tag.")

        # Check if either <message> or <finalize> tag is present, but not both
        has_message = "<message>" in response and "</message>" in response
        has_finalize = "<finalize>" in response and "</finalize>" in response

        if (state["turn"] > state["max_turns"]-2) and not has_finalize:
            errors.append("You must finalize before the turn limit!")

        if has_message and has_finalize:
            errors.append(
                "Response contains both <message>...</message> and <finalize>...</finalize> tags. Only one is allowed."
            )
        elif not has_message and not has_finalize:
            errors.append(
                "Response must contain either <message>...</message> or <finalize>...</finalize> tag."
            )

        # Ensure the player finalizes if the other player has already finalized
        if state["has_finalized"] and not has_finalize:
            errors.append(
                "The other player has made a finalization. You must finalize as well."
            )

        # Process finalization content if present
        if has_finalize:
            finalize_content = (
                response.split("<finalize>")[1].split("</finalize>")[0].strip()
            )

            try:
                finalize_json = json.loads(finalize_content)
                if not isinstance(finalize_json, dict): 
                    errors.append("The content within <finalize>...</finalize> is not a valid dictionary.")
                i_take = finalize_json.get("i_take")
                other_player_gets = finalize_json.get("other_player_gets")

                # Validate that the keys "i_take" and "other_player_gets" exist and have correct formats
                if not isinstance(i_take, dict) or not isinstance(
                    other_player_gets, dict
                ):
                    errors.append(
                        'The "i_take" and "other_player_gets" must be dictionaries.'
                    )
                elif not all(
                    isinstance(i_take.get(item), int) for item in state["items"]
                ) or not all(
                    isinstance(other_player_gets.get(item), int)
                    for item in state["items"]
                ):
                    errors.append(
                        'Each item in "i_take" and "other_player_gets" must be integers for game items.'
                    )

            except json.JSONDecodeError:
                errors.append(
                    "The content within <finalize>...</finalize> is not valid JSON."
                )

        # If there are errors, return with error message
        if errors:
            return True, "Errors: " + "; ".join(errors), False, None

        # If it's a valid finalization, return the parsed finalization data
        if has_finalize:
            return (
                False,
                "",
                True,
                {"i_take": i_take, "other_player_gets": other_player_gets},
            )

        # Extract and return message content if present
        if has_message:
            message_content = (
                response.split("<message>")[1].split("</message>")[0].strip()
            )
            return False, "", False, message_content

        # If neither valid message nor finalization is found
        return True, "Unknown error: Invalid response format.", False, None

            
    def set_game_info(self, state):
        """
        Gathers information from all rounds played and 
        adds it to the beginning of the augmented_context.

        Args:
            state (dict): The current state of the game.
        """
        round_infos = {
            "agreement_reached": [],
            "agreement_percentage": [],
            "self_points": [],
            "other_points": [],
            "imbalance": [],
            "ultimatum_ratio": [],
            "ultimatum_percentage": [],
            "max_ultimatum_points": [],
            "points_difference": [],  # New statistic
            "items_given_to_self": [],  # New statistic
            "self_points_on_agreement": [],  # New statistic
            "other_points_on_agreement": [],  # New statistic
            "points_diff_on_agreement": [],  # New statistic
        }

        round_infos_extra = {
            "quantities": [],
            "values": [],
        }
        
        for i in range(len(state['round_points'])):
            role = state['round_player_roles'][i][self.player_name]
            other_role = next(
                r for r in state['round_player_roles'][i].values() if r != role
            )
            self_points = state['round_points'][i][role]
            other_points = state['round_points'][i][other_role]
            
            round_infos["agreement_reached"].append(True if state['round_agreements_reached'][i] else False)
            round_infos["agreement_percentage"].append(100 if state['round_agreements_reached'][i] else 0)
            round_infos["self_points"].append(self_points)
            round_infos["other_points"].append(other_points)
            round_infos["points_difference"].append(self_points - other_points)  # Calculate the difference
            round_infos_extra["quantities"].append(state['round_quantities'][i])
            round_infos_extra["values"].append(state['round_values'][i][role])
            max_points = sum(state['round_quantities'][i][item] * state['round_values'][i][role][item] for item in state['round_quantities'][i].keys())
            max_ultimatum_points = max_points - min(state['round_values'][i][role].values())
            round_infos["max_ultimatum_points"].append(max_ultimatum_points)
            
            # Check for division by zero
            if max_ultimatum_points == 0:
                ultimatum_ratio = 0
            else:
                ultimatum_ratio = self_points / max_ultimatum_points
            round_infos["ultimatum_ratio"].append(ultimatum_ratio)
            
            ultimatum_percentage = 100 if ultimatum_ratio == 1.0 else 0
            round_infos["ultimatum_percentage"].append(ultimatum_percentage)
            
            # Check for division by zero
            total_points = self_points + other_points
            if total_points == 0:
                imbalance = 0
            else:
                imbalance = abs((self_points - other_points) / total_points)
            round_infos["imbalance"].append(imbalance)

            # New statistics
            if state['round_agreements_reached'][i]:
                round_infos["items_given_to_self"].append(state['round_finalizations'][i][role])
                round_infos["self_points_on_agreement"].append(self_points)
                round_infos["other_points_on_agreement"].append(other_points)
                round_infos["points_diff_on_agreement"].append(self_points - other_points)
            else:
                round_infos["items_given_to_self"].append(np.nan)
                round_infos["self_points_on_agreement"].append(np.nan)
                round_infos["other_points_on_agreement"].append(np.nan)
                round_infos["points_diff_on_agreement"].append(np.nan)
        
        # Correctly iterate over augmented_context to find and update 'round_info'
        c = 0
        for j in range(len(self.augmented_context)):
            if self.augmented_context[j]['role'] == 'round_info':
                content = {key: round_infos[key][c] for key in round_infos.keys()}
                content_extra = {key: round_infos_extra[key][c] for key in round_infos_extra.keys()}
                self.augmented_context[j] = {"role": "round_info", "content": content, "content_extra": content_extra}
                c += 1

        # Calculate mean for each metric and insert at the beginning of augmented_context
        content = round_infos
        game_info = {"role": "game_info", "content": content}
        self.augmented_context.insert(0, game_info)
        

    def set_preplay_round_info(self, state):
        round_info = {"role": "round_info", "content": {}}
        self.augmented_context.append(round_info)

    def new_round(self):
        """
        Resets round attributes.
        """
        self.retries = 0
        self.error_message = None

    def new_game(self, checkpoint=None):
        """
        Resets the message history of the LLM player or to a checkpoint if provided.

        Args:
            checkpoint (dict, optional): A dictionary containing the checkpoint state.
        """
        if checkpoint:
            self.load_checkpoint(checkpoint)
        else:
            self.retries = 0
            self.error_message = None
            self.context = []
            self.augmented_context = []

    def load_checkpoint(self, checkpoint):
        """
        Loads the player state from a checkpoint.

        Args:
            checkpoint (dict): A dictionary containing the checkpoint state.
        """
        self.__dict__.update(checkpoint)

def create_play_prompt(state):
    """
    Creates a play prompt based on the current game state.

    Args:
        state (dict): The current state of the game.

    Returns:
        str: The constructed play prompt or finalization prompt.
    """
    if state["has_finalized"]:
        return create_finalization_prompt(state)
    elif state["last_message"] is None:
        return "You are the first to play:\n"
    else:
        return f"The other player said: <QUOTE> {state['last_message']} </QUOTE>\n" 


def create_game_intro_prompt(state):
    """
    Creates a game introduction prompt for the Deal-or-No-Deal game.

    Args:
        state (dict): The current state of the game.

    Returns:
        str: The formatted game introduction prompt.
    """
    nb_rounds = state.get("nb_rounds", 1)
    max_turns = state.get("max_turns", 1)
    quantities = state.get("quantities", {})
    values = state["role_values"][state["player_to_role"][state["current_player"]]]
    game_mode_specificities = "Specific rules or conditions for the game mode."

    common_intro = f"""
    You will be playing {nb_rounds} rounds of a game called deal-or-no-deal.

    Deal-or-no-deal is a two-player negotiation game. I, the user, will be the game coordinator, acting as a middleman between you and the other player. Your objective is to maximize your personal points by proposing how to divide a set of item categories. All item categories must be distributed between you and the other player, and no items should be left over. The other player also aims to maximize their own points, which may or may not align with your interests.

    In this game, two players attempt to divide item categories between themselves, and each player may value the categories differently.

    {game_mode_specificities}
    """

    if max_turns == 1:
        # Special prompt for when max_turns is 1, without mentioning the turn limit
        prompt = f"""
        {common_intro}

        Game Mechanics:
        You can only send a final division. The final division should specify how many of each item category you want, leaving the remaining items for the other player. It should be JSON parsable.
        Matching Divisions: If the combined division doesn't match the total number of items available, both players score 0.

        Formatting:
        Final division: <finalize>{{ "i_take": {{"item_category1": x, "item_category2": y}}, "other_player_gets": {{"item_category1": y, "item_category2": x}} }}</finalize>, where 'i_take' is your share and 'other_player_gets' is the other player's share of the item categories.

        Example:
        1. You send:
        <finalize>{{ "i_take": {{"item_category1": x, "item_category2": y}}, "other_player_gets": {{"item_category1": y, "item_category2": x}} }}</finalize>

        2. The other player sends:
        <finalize>{{ "i_take": {{"item_category1": y, "item_category2": x}}, "other_player_gets": {{"item_category1": x, "item_category2": y}} }}</finalize>

        The first round starts now.
        Please decide how to divide {quantities} between yourself and the other player.
        To you, the item categories are worth: {values}.
        """
    else:
        # Standard prompt for when max_turns is greater than 1
        prompt = f"""
        {common_intro}

        Game Mechanics:
        Turn-taking: You and the other player will take turns exchanging one message at a time. After enough exchanges, when you feel ready, you can finalize the negotiation by sending the division to the game coordinator. Once a player decides to send a final division, the other player must also send a final division, ending the game.
        Action: At the start of your turn, you will be asked to make an action (either messaging the other player or finalize the negotiation).
        Final Division: The final division should specify how many of each item category you want, leaving the remaining items for the other player. It should be JSON parsable.
        Matching Divisions: If the combined division doesn't match the total number of items available, both players score 0.
        There is a limit of 40 characters per message.

        Formatting:
        Messages: <message> [Your message here.] </message>
        Final division: <finalize>{{ "i_take": {{"item_category1": 0, "item_category2": 0}}, "other_player_gets": {{"item_category1": 0, "item_category2": 0}} }}</finalize>, where 'i_take' is your share and 'other_player_gets' is the other player's share of the item categories.

        Only do one action per turn.

        Examples of how turns might proceed:
        1. [Initial state is given]
        <message> [Your message to the other player here.] </message>

        2. [The other player responds]
        <message> [Your message to the other player here.] </message>

        3. [The other player agrees]
        <finalize>{{ "i_take": {{"item_category1": 0, "item_category2": 0}}, "other_player_gets": {{"item_category1": 0, "item_category2": 0}} }}</finalize>

        The first round starts now.
        Please decide how to divide {quantities} between yourself and the other player.
        To you, the item categories are worth: {values}.
        """
    return prompt.strip()


def create_finalization_prompt(state):
    """
    Creates a finalization prompt for the Deal-or-No-Deal game.

    Args:
        state (dict): The current state of the game.

    Returns:
        str: The formatted finalization prompt.
    """
    quantities = state.get("quantities", {})
    values = state.get("values", {})
    other_player_finalization = state.get("last_message", "")

    prompt = f"""
    A finalization has been made by the other player. It's your turn to finalize the division of items.

    Please ensure that no items are left over. Your finalization should be formatted as follows:
    <finalize>{{ "i_take": {{"item_category_1": x, "item_category_2": y}}, "other_player_gets": {{"item_category_1": y, "item_category_2": x}} }}</finalize>

    Here, 'i_take' represents your share of the items, and 'other_player_gets' represents the other player's share.

    Remember:
    - All items must be distributed between you and the other player.
    - The total number of items in 'i_take' and 'other_player_gets' should match the available quantities: {quantities}.
    - To you, the items are worth: {values}.
    """

    if state.get("finalization_visibility", False) and other_player_finalization:
        prompt += f"\nAs a clue, the other player's finalization was: {other_player_finalization}\n"

    prompt += "\nPlease make your finalization decision now."
    
    return prompt.strip()


def create_new_round_prompt(state):
    """
    Creates a new round prompt including the outcome of the last round.

    Args:
        state (dict): The current state of the game.

    Returns:
        str: The constructed new round prompt.
    """
    agreement_reached = state.get(['round_agreements_reached'][-1], False)
    current_round = state.get("round_number", 1)
    nb_rounds = state.get("nb_rounds", 1)
    quantities = state.get("quantities", {})
    values = state["role_values"][state["player_to_role"][state["current_player"]]]
    self_points = state['round_points'][-1][state["player_to_role"][state["current_player"]]]

    last_round_info = (
        f"An agreement was reached in the last round.\n"
        f"You scored {self_points} points."
        if agreement_reached else "No agreement was reached in the last round."
    )

    return (
        f"Last round info: {last_round_info}\n"
        f"You are now playing round {current_round+1}/{nb_rounds}.\n"
        f"For this round, the quantities are {quantities}"
        f"To you, the items are worth {values}"
    )




