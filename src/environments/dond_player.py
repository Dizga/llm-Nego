import json
import regex as re
import copy
# local imports
from environments.dond_game import DondGame
import math
from statistics import mean
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

        if state["is_new_round"]:
            self.new_round()
            self.set_preplay_round_info(state)

        if state["is_new_game"]:
            user_message += self.game_basics.format(**state)

        if state["is_new_round"] and not state["is_new_game"]:
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
        round_infos = {
            "agreement_reached": [],
            "agreement_percentage": [],
            "self_points": [],
            "other_points": [],
            "imbalance": [],
            "ultimatum_ratio": [],
            "ultimatum_percentage": [],
            "max_ultimatum_points": [],
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
            round_infos["agreement_reached"].append(True if state['round_agreements_reached'][i] else False)
            round_infos["agreement_percentage"].append(100 if state['round_agreements_reached'][i] else 0)
            round_infos["self_points"].append(state['round_points'][i][role])
            round_infos["other_points"].append(state['round_points'][i][other_role])
            round_infos_extra["quantities"].append(state['round_quantities'][i])
            round_infos_extra["values"].append(state['round_values'][i][role])
            max_points = sum(state['round_quantities'][i][item] * state['round_values'][i][role][item] for item in state['round_quantities'][i].keys())
            max_ultimatum_points = max_points - min(state['round_values'][i][role].values())
            round_infos["max_ultimatum_points"].append(max_ultimatum_points)
            
            # Check for division by zero
            if max_ultimatum_points == 0:
                ultimatum_ratio = 0
            else:
                ultimatum_ratio = state['round_points'][i][role] / max_ultimatum_points
            round_infos["ultimatum_ratio"].append(ultimatum_ratio)
            
            ultimatum_percentage = 100 if ultimatum_ratio == 1.0 else 0
            round_infos["ultimatum_percentage"].append(ultimatum_percentage)
            
            # Check for division by zero
            total_points = state['round_points'][i][role] + state['round_points'][i][other_role]
            if total_points == 0:
                imbalance = 0
            else:
                imbalance = abs((state['round_points'][i][role] - state['round_points'][i][other_role]) / total_points)
            round_infos["imbalance"].append(imbalance)
        
        # Correctly iterate over augmented_context to find and update 'round_info'
        c = 0
        for j in range(len(self.augmented_context)):
            if self.augmented_context[j]['role'] == 'round_info':
                content = {key: round_infos[key][c] for key in round_infos.keys()}
                content_extra = {key: round_infos_extra[key][c] for key in round_infos_extra.keys()}
                self.augmented_context[j] = {"role": "round_info", "content": content, "content_extra": content_extra}
                c += 1

        # Calculate mean for each metric and insert at the beginning of augmented_context
        content = {key: mean(round_infos[key]) for key in round_infos.keys()}
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



