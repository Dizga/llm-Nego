import json
import regex as re
import copy
# local imports
from environments.dond_game import DondGame


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
        model_name,
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
            model_name (str): The name of the model used.
        """
        with open(game_intro_file, "r") as file:
            self.game_basics = file.read()

        with open(new_round_file, "r") as file:
            self.new_round_prompt = file.read()

        with open(finalization_file, "r") as file:
            self.finalization_prompt = file.read()
    
        with open(in_between_file, "r") as file:
            self.in_between_prompt = file.read()

        self.allow_reasoning = allow_reasoning
        self.player_name = player_name
        self.max_retries = max_retries
        self.model_name = model_name
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

        return send_to_game, is_finalization, processed_response


    def set_usr_message(self, state):
        """
        Constructs a user message based on the current game state.

        Args:
            state (dict): The current state of the game.

        Returns:
            str: The constructed user message.
        """

        # Create dummy finalization to include in game explanation
        dummy_finalization = {key: "..." for key in state["quantities"]}
        state["dummy_finalization"] = json.dumps(dummy_finalization)
        state["game_mode_specificities"] = self.game_state_specificities(state["mode"])
        state['values'] = state['role_values'][state['player_to_role'][self.player_name]]

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

        if state["is_new_game"]:
            self.set_round_info(state, post_round=False)
            user_message += self.game_basics.format(**state)

        if state["is_new_round"] and not state["is_new_game"]:
            self.new_round()
            self.set_round_info(state, post_round=False)
            user_message += self.new_round_prompt.format(**state)

        if state["has_finalized"]:
            user_message += self.finalization_prompt.format(**state)
            if state["finalization_visibility"]:
                user_message += (
                    f"As a clue, the other player's proposal was: '{state['last_message']}'\n"
                )

        if state["last_message"] is None:
            user_message += "You are the first to play, there are no messages yet.\n"

        elif not state["has_finalized"]:
            user_message += f"The other player said: '{state['last_message']}'\n"
            user_message += self.in_between_prompt.format(**state)

        usr_prompt = {
            "role": "user",
            "content": user_message,
            "is_error": False,
            "round_nb": state["round_number"],
        }
        self.add_to_context(usr_prompt)

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

    def game_state_specificities(self, mode):
        """
        Retrieves game mode specific instructions.

        Args:
            mode (str): The game mode.

        Returns:
            str: The specific instructions for the given game mode.
        """
        if mode == "basic":
            with open("src/prompts/basic.txt", "r") as file:
                return file.read()
        if mode == "coop":
            with open("src/prompts/coop.txt", "r") as file:
                return file.read()
            
            
    def set_game_info(self, state):
        """
        Gathers information from all rounds played and 
        adds it to the beginning of the augmented_context.

        Args:
            state (dict): The current state of the game.
        """
        game_info = {
            "role": "game_info",
            "content": {
                "game_agreement_rate": sum(state["round_agreements"]) / (len(state["round_agreements"]) + 10e-10),
                "game_self_points": sum(state["round_points"][self.player_name]),
                #"game_other_points": sum(state["round_points"][state["current_turn"]]),
                "round_points": state["round_points"],
                "round_agreements": state["round_agreements"],
                "total_rounds": state["nb_rounds"],
                "completed_rounds": state["round_number"] - 1,
            }
        }

        # Insert the game_info at the beginning of the augmented_context
        self.augmented_context.insert(0, game_info)

    def set_round_info(self, state, post_round=True):
        """
        Adds information about the round at the beginning of the round exchange.

        Args:
            state (dict): The current state of the game.
            post_round (bool): If True, updates the round info with end-of-round details.
        """
        if not post_round:
            self_role = state['player_to_role'][self.player_name]
            round_info = {
                "role": "round_info",
                "content": {
                    "quantities": state["quantities"],
                    "values_self": state["role_values"][self_role],
                    #"values_other": state["role_values"][other_role],
                    "round_number": state["round_number"],
                },
            }
            self.augmented_context.append(round_info)

        else:
            for info in self.augmented_context:
                if (
                    info["role"] == "round_info"
                    and info["content"]["round_number"] == state["round_number"]-1
                ):
                    info["content"].update(
                        {
                            "agreement_reached": state["round_agreements"][-1],
                            "round_points": state["round_points"][self.player_name][-1],
                        }
                    )
                    break

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