import json
import regex as re
import copy
# local imports
from environments.dond.dond_game import DondGame
import math
from statistics import mean
import numpy as np

class DondPlayerHandler:
    def __init__(
        self,
        player_name,
        allow_reasoning,
        max_retries,
        mod_adpt_id,
    ):
        """
        Initializes the DondPlayerHandler.

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
        self.reset()

    def get_chat_history(self):
        return self.chat_history

    def add_to_chat_history(self, element: dict):
        self.chat_history.append(element)

    def step(self, input):
        """
        Processes the response from the model and updates the game state.

        Args:
            action (str): The action to be taken.
            state (dict): The current state of the game.
            llm_output (str): The output from the language model.

        Returns:
            tuple: A tuple containing:
                - observation (dict): The new state of the game.
                - reward (float): The reward obtained from the action.
                - done (bool): Whether the game is finished.
                - info (dict): Additional information.
        """
        state, info, llm_output = input
        # Initiate what will be returned
        processed_response = None
        send_to_game = False
        is_finalization = False

        # Process response. Check for errors.
        is_error, error_message, is_finalization, processed_response = self.process_response(
            llm_output, state
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

        # Add raw response to chat_history
        model_response = {
            "role": "assistant",
            "content": llm_output,
            "is_error": is_error,
            "is_finalization": is_finalization,
            "round_nb": state["round_number"],
        }

        self.add_to_chat_history(model_response)
        
        action = (is_finalization, processed_response)
        player_state = None
        player_info = {"player_name": self.player_name, "chat_history": self.chat_history}

        return action, player_state, send_to_game, player_info
    def get_info(self):
        return {"player_name": self.player_name, "chat_history": self.chat_history}
    
    # Optional render method
    def render(self, mode='human'):
        """
        Renders the environment for visualization.
        """
        # Implement rendering logic if needed
        pass

    # Optional close method
    def close(self):
        """
        Cleans up resources when the environment is no longer needed.
        """
        # Implement cleanup logic if needed
        pass

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
            self.add_to_chat_history(usr_prompt)
            self.error_message = None
            return
        
        if state["is_new_game"]:
            user_message += create_game_intro_prompt(state)

        if state["is_new_round"]:
            self.new_round()
            user_message += create_new_round_prompt(state)


        user_message += create_play_prompt(state)

        user_message = {
            "role": "user",
            "content": user_message,
            "is_error": False,
            "round_nb": state["round_number"],
        }
        self.add_to_chat_history(user_message)

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
                    i_take = None
                    other_player_gets = None
                else:
                    i_take = finalize_json.get("i_take", {})
                    other_player_gets = finalize_json.get("other_player_gets", {})

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


    def new_round(self):
        """
        Resets round attributes.
        """
        self.retries = 0
        self.error_message = None

    def reset(self, checkpoint=None):
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
            self.chat_history = []
            self.augmented_chat_history = []
        return self.chat_history  # Return initial observation

    def load_checkpoint(self, checkpoint):
        """
        Loads the player state from a checkpoint.

        Args:
            checkpoint (dict): A dictionary containing the checkpoint state.
        """
        self.__dict__.update(checkpoint)

    def export(self, path, state_history):
        game_stats = self.gather_statistics_func(state_history, self.conversation_history, **self.gather_statistics_func_args)
        self.set_chat_returns_func(self.conversation_history, **self.set_chat_returns_func_args)

        with open(path, "w") as f:
            json.dump(game_stats, f)
            json.dump(self.conversation_history, f)


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
    Constructs a game introduction prompt.

    Args:
        state (dict): The current state of the game.

    Returns:
        str: The formatted game introduction prompt.
    """
    nb_rounds = state.get("nb_rounds", 1)
    max_turns = state.get("max_turns", 1)
    game_mode_specificities = "Specific rules or conditions for the game mode."

    common_intro = f"""
    Welcome to the splitting game. 
    You will engage in {nb_rounds} rounds of splitting, where two players aim to divide items from various categories. 

    Each player may assign different values to these categories, and your primary objective is to maximize your personal cumulative points.

    Points are determined at the end of each round by multiplying the quantity of items you acquire by their respective values. Your cumulative points across all rounds will determine your success. Note that the other player will also strive to maximize their points, which may not align with your interests.

    Importantly, in the event that no agreement is reached within a round, both players will receive zero points.

    {game_mode_specificities}
    """

    if max_turns == 1:
        # Special prompt for when max_turns is 1, without mentioning the turn limit
        prompt = f"""
        {common_intro}

        Game Mechanics:
        
        You are required to submit a final division of items. This division should clearly specify the quantity of each item category you wish to claim, with the remainder allocated to the other player. The division must be in a JSON-parsable format.
        
        Matching Divisions: If the combined division does not correspond to the total number of available items, both players will score zero points.

        Formatting:
        
        Final division: <finalize>{{ "i_take": {{"item_category1": x, "item_category2": y}}, "other_player_gets": {{"item_category1": y, "item_category2": x}} }}</finalize>, where 'i_take' represents your share and 'other_player_gets' represents the other player's share of the item categories.

        Example:
        
        1. You submit:
        <finalize>{{ "i_take": {{"item_category1": x, "item_category2": y}}, "other_player_gets": {{"item_category1": y, "item_category2": x}} }}</finalize>

        2. The other player submits:
        <finalize>{{ "i_take": {{"item_category1": y, "item_category2": x}}, "other_player_gets": {{"item_category1": x, "item_category2": y}} }}</finalize>
        """
    else:
        # Standard prompt for when max_turns is greater than 1
        prompt = f"""
        {common_intro}

        Game Mechanics:
        
        Turn-taking: You and the other player will alternate turns, exchanging one message at a time. When you are ready, you may finalize the negotiation by submitting your division. Once a player decides to finalize, the other player must also submit their final division, concluding the game.
        
        Action: At the start of your turn, you will be prompted to take an action, either by messaging the other player or finalizing the negotiation.
        
        Final Division: The final division should specify the quantity of each item category you wish to claim, with the remainder allocated to the other player. The division must be in a JSON-parsable format.
        
        Matching Divisions: If the combined division does not correspond to the total number of available items, both players will score zero points.
        
        There is a limit of 40 characters per message.

        Formatting:
        
        Messages: <message> [Your message here.] </message>
        
        Final division: <finalize>{{ "i_take": {{"item_category1": 0, "item_category2": 0}}, "other_player_gets": {{"item_category1": 0, "item_category2": 0}} }}</finalize>, where 'i_take' represents your share and 'other_player_gets' represents the other player's share of the item categories.

        Only one action is permitted per turn.

        Examples of turn progression:
        
        1. [Initial state is provided]
        <message> [Your message to the other player here.] </message>

        2. [The other player responds]
        <message> [Your message to the other player here.] </message>

        3. [The other player agrees]
        <finalize>{{ "i_take": {{"item_category1": 0, "item_category2": 0}}, "other_player_gets": {{"item_category1": 0, "item_category2": 0}} }}</finalize>
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
    other_player_finalization = state.get("last_message", "")

    prompt = f"""
    A finalization has been made by the other player. It's your turn to finalize the division of items.

    Your finalization should be formatted as follows:
    <finalize>{{ "i_take": {{"item_category_1": x, "item_category_2": y}}, "other_player_gets": {{"item_category_1": y, "item_category_2": x}} }}</finalize>

    Here, 'i_take' represents your share of the items, and 'other_player_gets' represents the other player's share.

    Remember:
    - The total number of items in 'i_take' and 'other_player_gets' should match the available quantities: {quantities}.
    - 0 points to each player if the division is not matching
    """

    if state.get("finalization_visibility", False) and other_player_finalization:
        prompt += (
            f"\nAs a clue, the other player's finalization was for you to take: "
            f"{other_player_finalization['other_player_gets']} and for them to take: "
            f"{other_player_finalization['i_take']}.\n"
        )

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
    current_round = state.get("round_number", 1)
    nb_rounds = state.get("nb_rounds", 1)
    quantities = state.get("quantities", {})
    values = state["role_values"][state["player_to_role"][state["current_player"]]]

    if current_round > 0:
        agreement_reached = state['round_agreements_reached'][-1] if state['round_agreements_reached'] else False
        self_points = state['round_points'][-1][state["player_to_role"][state["current_player"]]]

        last_round_info = (
            f"An agreement was reached in the last round.\n"
            f"You scored {self_points} points."
        ) if agreement_reached else "No agreement was reached in the last round."
        last_round_info = f"Last round info: {last_round_info}\n"
    else:
        last_round_info = ""

    # New addition: Check for finalization visibility and include other player's values
    other_player_values = ""
    if state.get("other_values_visibility", False):
        other_role = next(role for role in state["role_values"] if role != state["player_to_role"][state["current_player"]])
        other_player_values = f"\nTo the other player, the items are worth {state['role_values'][other_role]}."

    return (
        f"{last_round_info}"
        f"You are now playing round {current_round+1}/{nb_rounds}.\n"
        f"For this round, the quantities are {quantities}.\n"
        f"To you, the items are worth {values}.{other_player_values}"
    )




