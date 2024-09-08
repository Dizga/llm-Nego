import json
import regex as re
import copy
# local imports
from environments.dond_game import DondGame


class DondPlayer():
    def __init__(self, 
                 player_name,
                 game_intro_file, 
                 chain_of_thought_file, 
                 new_round_file,
                 max_retries,
                 finalization_file,
                 game_state,
                 model_name):
        """
        The Player acts as a middle-man between the game and a LLM player.
        Initializes the DoNDPlayer.

        Args:
            game_intro_file (str): Path to the file containing game introduction.
            chain_of_thought_file (str): Path to the file containing chain of thought instructions.
            dond_game (DoND): The DoND game instance.
            agent (NegoAgent): The LLM player instance.
            player_type (str): The type of player, either "player_0" or "player_1".
        """
        self.first_move = True
        self.is_new_game = True

        # Pattern to match the message part
        self.message_pattern = r'<message>(.*?)</message>'
        
        # Pattern to match the finalization part
        self.finalize_pattern = r'<finalize>\s*\{\s*"i_take"\s*:\s*(.*?)\s*,\s*"other_player_gets"\s*:\s*(.*?)\s*\}\s*</finalize>'
        
        with open(game_intro_file, 'r') as file:
            self.game_basics = file.read()

        with open(new_round_file, 'r') as file:
            self.new_round_prompt = file.read()

        with open(finalization_file, 'r') as file:
            self.finalization_prompt = file.read()
        
        self.chain_of_thought = None
        if chain_of_thought_file:
            with open(chain_of_thought_file, 'r') as file:
                self.chain_of_thought = file.read()

        self.player_name = player_name
        self.round_nb = 1
        self.retries = 0
        self.max_retries = max_retries
        self.model_name = model_name
        self.other_has_finalized = False  # whether the other player has made a finalization
        self.error_overload_message = False
        self.context = []
        self.reset_game(game_state)



    def get_context(self):
        return self.context
    
    def add_to_context(self, element: dict):
        self.context.append(element)


    def process_model_response(self, response, state):
        """
        
        """

        # Initiate what will be returned
        processed_response = None
        send_to_game = False
        is_finalization= False

        # Verify if model response was valid
        is_error, self.error_message = self.validate(response, state)

        if is_error: 
            self.retries += 1
            # Too many mistakes were made
            if self.retries > self.max_retries:
                response = "<reason></reason><message>I have failed to provide a proper response.</message>"
                self.error_overload_message = f"""Last turn, you made too many errors. 
                The final one was: "{self.error_message}". 
                The dummy response "{response}" was sent to the other player in place of the one you sent."""
                send_to_game = True

        else: 
            self.retries = 0
            send_to_game = True
            is_finalization, processed_response = self.extract(response)

        # Add raw response to context
        model_response = {'role': 'assistant', 'content': response, 'is_error': is_error, 'is_finalization': is_finalization, 'is_new_round': self.is_new_round}
        self.add_to_context(model_response)

        if not is_finalization: # Add user response to context
            self.set_usr_message(state, is_error=is_error)

        return processed_response, send_to_game, is_finalization


    def set_usr_message(self, state, is_error=False):
        """
        Constructs a user message based on the current game state.

        Args:
            state (dict): The current state of the game.

        Returns:
            str: The constructed user message.
        """

        # Create dummy finalization to include in game explanation
        dummy_finalization = {key: "..." for key in state['quantities']}
        state['dummy_finalization'] = dummy_finalization
        state['game_mode_specificities'] = self.game_state_specificities(state['mode'])

        user_message = ""

        if is_error:
            user_message = self.error_message
            usr_prompt = {'role': 'user', 'content': user_message, 'is_error': is_error, 'is_new_round': False}
            self.add_to_context(usr_prompt) 
            return

        if self.error_overload_message:
            user_message += self.error_overload_message
            usr_prompt = {'role': 'user', 'content': user_message, 'is_error': is_error, 'is_new_round': False}
            self.add_to_context(usr_prompt) 
            self.error_overload_message = False
            return

        # if state['round_number'] > self.round_nb:
        #     self.reset_round()
        #     user_message += self.new_round_prompt.format(**state)
        #     self.round_nb+=1
        #     self.is_new_round = True
        # else: self.is_new_round = False

        if self.is_new_game:
            user_message += self.game_basics.format(**state)
            self.is_new_game = False

        if state["has_finalized"]:
            self.other_has_finalized = True
            user_message += self.finalization_prompt.format(**state)

        if state['last_message'] == None:
            user_message += "You are the first to play, there are no messages yet.\n"

        else:
            user_message += f"The other player said: '{state['last_message']}'\n"

        if self.chain_of_thought is not None:
                user_message += self.chain_of_thought.format(**state)

        usr_prompt = {'role': 'user', 'content': user_message, 'is_error': is_error, 'is_new_round': self.is_new_round}
        self.add_to_context(usr_prompt) 

    
    def game_state_specificities(self, mode):
        if mode == "basic":
            with open('src/prompts/basic.txt', 'r') as file:
                return file.read()
        if mode == "coop":
            with open('src/prompts/coop.txt', 'r') as file:
                return file.read()


    
    def validate(self, response, state):
        """
        Validates the response from the LLM player.

        Args:
            response (str): The response from the LLM player.

        Returns:
            tuple: A tuple containing a boolean indicating if error is present and an error message if invalid.
        """
        errors = []
        
        # Check if reasoning tag exists
        if "<reason>" not in response or "</reason>" not in response:
            errors.append("Missing <reason>...</reason> tag.")
        
        # Check if message or finalize tag exists, but not both
        has_message = "<message>" in response and "</message>" in response
        has_finalize = "<finalize>" in response and "</finalize>" in response
        
        if has_message and has_finalize:
            errors.append("Response contains both <message>...</message> and <finalize>...</finalize> tags. Only one is allowed per response.")

        elif not has_message and not has_finalize:
            errors.append("Response must contain either <message>...</message> or <finalize>...</finalize> tag. Do not forget the closing tag.")

        if self.other_has_finalized:
            if not has_finalize:
                errors.append("The other player has made a finalization. You must finalize also.")
        
        # Check if finalize tag is JSON parsable and follows the specified format
        if has_finalize:

            finalize_content = response.split("<finalize>")[1].split("</finalize>")[0].strip()

            try:

                finalize_json = json.loads(finalize_content)

                if not all(key in finalize_json for key in ["i_take", "other_player_gets"]):
                    errors.append('The <finalize> tag must contain JSON with keys "i_take" and "other_player_gets".')

                else:
                    # TODO: use json dict catcher instead!
                    i_take = finalize_json["i_take"]
                    other_player_gets = finalize_json["other_player_gets"]
                    # Generalized validation for arbitrary items

                    # Does not self-attribute right set of items for self
                    if not (isinstance(i_take, dict) and 
                            all(key in i_take for key in state['items']) and 
                            all(isinstance(i_take[key], int) for key in state['items'])):
                        errors.append('The "i_take" value must be a dictionary with integer values for the game items.')
                    
                    # Does not attribute right set of items for other player
                    if not (isinstance(other_player_gets, dict) and 
                            all(key in other_player_gets for key in state['items']) and 
                            all(isinstance(other_player_gets[key], int) for key in state['items'])):
                        errors.append('The "other_player_gets" value must be a dictionary with integer values for the game items.')


                    if not re.search(self.finalize_pattern, response, re.DOTALL):
                        errors.append('Could not pattern match on finalization.')


            except json.JSONDecodeError:
                errors.append("The content within <finalize>...</finalize> is not valid JSON.")

        # Generate error message or return success
        if errors:
            return True, "Errors: " + "; ".join(errors)
        else:
            return False, "Response is valid."


    def extract(self, response):
        """
        Extracts the content from the response.

        Args:
            response (str): The full response string.

        Returns:
            tuple: A tuple containing a boolean indicating if it's a finalization 
                and the extracted content (either a string message or a dictionary 
                for finalization details).
        """
        
        # Check if it's a message
        message_match = re.search(self.message_pattern, response, re.DOTALL)
        
        if message_match:
            # Extract and return the message content
            message_content = message_match.group(1)
            return False, message_content
        
        # Check if it's a finalization
        finalize_match = re.search(self.finalize_pattern, response, re.DOTALL)
        
        if finalize_match:
            # Extract the finalization data and convert it to JSON
            i_take_json = finalize_match.group(1)
            other_player_gets_json = finalize_match.group(2)
            
            i_take = json.loads(i_take_json)
            other_player_gets = json.loads(other_player_gets_json)
            
            # Return the finalization data
            return True, {"i_take": i_take, "other_player_gets": other_player_gets}
        
        # If neither message nor finalization is found, return False with an empty string
        return False, ""

        
    def reset_round(self):
        """
        Resets round attributes.
        """
        self.is_new_round = True
        self.other_has_finalized = False
        self.error_overload_message = False

    
    def reset_game(self, state):
        """
        Resets the message history of the LLM player.

        Returns:
            list: The message history before resetting.
        """
        self.reset_round()
        self.round_nb = 1
        self.is_new_game = True
        self.is_new_round = True
        self.first_move = True
        self.other_has_finalized = False
        self.error_overload_message = False
        self.context = []
        self.set_usr_message(state)

