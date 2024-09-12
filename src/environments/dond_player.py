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
        self.max_retries = max_retries
        self.model_name = model_name
        self.game_id = None # ID of player in game
        self.reset_game()


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

        # Process response. Check for errors.
        is_error, error_message, is_finalization, processed_response = self.process_response(response, state)

        if is_error: 
            self.retries += 1
            self.error_message = error_message
            # Too many mistakes were made
            if self.retries > self.max_retries:
                self.error_message = False
                response = "<reason></reason><message>I have made too many errors</message>"
                processed_response = "I have made too many errors"
                # self.error_message = f"""Last turn, you made too many errors. 
                # The final one was: "{self.error_message}". 
                # The dummy response "{response}" was sent to the other player in place of the one you sent."""
                send_to_game = True
                self.retries = 0

        else: 
            self.retries = 0
            send_to_game = True

        # Add raw response to context
        model_response = {'role': 'assistant', 
                          'content': response, 
                          'is_error': is_error, 
                          'is_finalization': is_finalization, 
                          'round_nb': state['round_number']
                          }
        
        self.add_to_context(model_response)

        return send_to_game, is_finalization, processed_response


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
        state['dummy_finalization'] = json.dumps(dummy_finalization)
        state['game_mode_specificities'] = self.game_state_specificities(state['mode'])

        user_message = ""

        if self.error_message:
            user_message = self.error_message
            usr_prompt = {'role': 'user', 
                          'content': user_message, 
                          'is_error': False, 
                          'round_nb': state['round_number']}
            self.add_to_context(usr_prompt) 
            self.error_message = None
            return

        if state["is_new_game"]:
            user_message += self.game_basics.format(**state)

        if state['is_new_round'] and not state["is_new_game"]:
            self.reset_round()
            user_message += self.new_round_prompt.format(**state)

        if state["has_finalized"]:
            self.other_has_finalized = True
            user_message += self.finalization_prompt.format(**state)

        if state['last_message'] == None:
            user_message += "You are the first to play, there are no messages yet.\n"

        else:
            user_message += f"The other player said: '{state['last_message']}'\n"

        if self.chain_of_thought is not None:
                user_message += self.chain_of_thought.format(**state)

        usr_prompt = {'role': 'user', 
                      'content': user_message, 
                      'is_error': is_error, 
                      'round_nb': state['round_number']
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

        # Check for presence of <reason> tag
        if "<reason>" not in response or "</reason>" not in response:
            errors.append("Missing <reason>...</reason> tag.")
        
        # Check if either <message> or <finalize> tag is present, but not both
        has_message = "<message>" in response and "</message>" in response
        has_finalize = "<finalize>" in response and "</finalize>" in response

        if has_message and has_finalize:
            errors.append("Response contains both <message>...</message> and <finalize>...</finalize> tags. Only one is allowed.")
        elif not has_message and not has_finalize:
            errors.append("Response must contain either <message>...</message> or <finalize>...</finalize> tag.")
        
        # Ensure the player finalizes if the other player has already finalized
        if state['has_finalized'] and not has_finalize:
            errors.append("The other player has made a finalization. You must finalize as well.")

        # Process finalization content if present
        if has_finalize:
            finalize_content = response.split("<finalize>")[1].split("</finalize>")[0].strip()
            
            try:
                finalize_json = json.loads(finalize_content)
                i_take = finalize_json.get("i_take")
                other_player_gets = finalize_json.get("other_player_gets")
                
                # Validate that the keys "i_take" and "other_player_gets" exist and have correct formats
                if not isinstance(i_take, dict) or not isinstance(other_player_gets, dict):
                    errors.append('The "i_take" and "other_player_gets" must be dictionaries.')
                elif not all(isinstance(i_take.get(item), int) for item in state['items']) or \
                    not all(isinstance(other_player_gets.get(item), int) for item in state['items']):
                    errors.append('Each item in "i_take" and "other_player_gets" must be integers for game items.')
            
            except json.JSONDecodeError:
                errors.append("The content within <finalize>...</finalize> is not valid JSON.")
        
        # If there are errors, return with error message
        if errors:
            return True, "Errors: " + "; ".join(errors), False, None

        # If it's a valid finalization, return the parsed finalization data
        if has_finalize:
            return False, "", True, {"i_take": i_take, "other_player_gets": other_player_gets}
        
        # Extract and return message content if present
        if has_message:
            message_content = response.split("<message>")[1].split("</message>")[0].strip()
            return False, "", False, message_content

        # If neither valid message nor finalization is found
        return True, "Unknown error: Invalid response format.", False, None

    def game_state_specificities(self, mode):
        if mode == "basic":
            with open('src/prompts/basic.txt', 'r') as file:
                return file.read()
        if mode == "coop":
            with open('src/prompts/coop.txt', 'r') as file:
                return file.read()
            
    def set_round_scores(self, state):
        for item in self.context:
            if item['role'] == 'assistant' and item['round_nb'] == state['round_number']-1:
                item['self_score'] = state['last_scores'][self.game_id]
                item['other_score'] = state['last_scores'][1-self.game_id]

    def reset_round(self):
        """
        Resets round attributes.
        """
        self.retries = 0
        self.error_message = None

    
    def reset_game(self):
        """
        Resets the message history of the LLM player.

        Returns:
            list: The message history before resetting.
        """
        self.retries = 0
        self.error_message = None
        self.context = []

