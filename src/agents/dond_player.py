import json
import regex as re
import copy
# local imports
from environments.dond_game import DondGame


class DondPlayer():
    def __init__(self, 
                 dond_game,
                 game_intro_file, 
                 chain_of_thought_file, 
                 new_round_file,
                 max_retries,
                 finalization_file,
                 agent, 
                 player_type="player_0"):
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

        self.round_nb = 1
        self.dond_game = dond_game
        self.max_retries = max_retries
        self.agent = agent
        self.player_type = player_type
        self.other_has_finalized = False  # whether the other player has made a finalization
        self.error_overload_message = False

    def play_move(self, state):
        """
        Plays a move in the DoND game.

        """

        # Check if new round
        if state['round_number'] > self.round_nb:
            self.new_round()
            self.round_nb+=1
        else: 
            self.is_new_round = False

        # Get the context message to be passed to the model to get its response
        user_message = self.get_usr_message(state)

        # Get response from the model
        response = self.agent.prompt(user_message, is_new_round=self.is_new_round)

        # Validate the response from the model
        valid_response, error_message = self.validate(response)

        # Allow safety nets which gives retry attempts to the model
        retries = 0
        while not valid_response and retries <= self.max_retries:
            response = self.agent.prompt(error_message, is_error=True)
            valid_response, error_message = self.validate(response)
            if not valid_response:
                self.agent.set_error_last_message() # Set error in agent history for last message
            retries += 1

        # Too many mistakes were made
        if not valid_response and retries > self.max_retries:
            response = "<reason></reason><message>I have failed to provide a proper response.</message>"
            self.error_overload_message = f"""Last turn, you made too many errors. The final one was: "{error_message}". The dummy response "{response}" was sent to the other player in place of the one you sent."""

        self.first_move = False

        # Process the response
        return self.extract(response)


    def get_usr_message(self, state):
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

        if self.is_new_game:
            user_message += self.game_basics.format(**state)
            self.is_new_game = False

        if self.is_new_round: 
            user_message += self.new_round_prompt.format(**state)

        if self.error_overload_message:
            user_message += self.error_overload_message
            self.error_overload_message = False

        if state["has_finalized"]:
            self.other_has_finalized = True
            user_message += self.finalization_prompt.format(**state)

        else:
            user_message += f"The other player said: '{state['last_message']}'\n" if state['last_message'] else "You are the first to play, there are no messages yet.\n"

            if self.chain_of_thought is not None:
                user_message += self.chain_of_thought.format(**state)

        return user_message
    
    def game_state_specificities(self, mode):

        if mode == "basic":
            return """
            You are playing the vanilla variation of this game.
            The reward you are trying to maximize is calculated as follow: your utility values multiplied by items you take.
            """
        
        if mode == "coop":
            return """
            You are playing the cooperative variation of the deal-or-no-deal game. 
            The reward you are trying to maximize is calculated as follows: your utility values multiplied by items you take + the other player's utility values multiplied by the items they take

            I repeat, it is the reward that you are trying to maximize and not only your utility values multiplied by items you take. It is the sum of both total utilities, per the cooperation mode.
            """


    
    def validate(self, response):
        """
        Validates the response from the LLM player.

        Args:
            response (str): The response from the LLM player.

        Returns:
            tuple: A tuple containing a boolean indicating validity and an error message if invalid.
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
                            all(key in i_take for key in self.dond_game.items) and 
                            all(isinstance(i_take[key], int) for key in self.dond_game.items)):
                        errors.append(f'The "i_take" value must be a dictionary with integer values for {self.dond_game.items}.')
                    
                    # Does not attribute right set of items for other player
                    if not (isinstance(other_player_gets, dict) and 
                            all(key in other_player_gets for key in self.dond_game.items) and 
                            all(isinstance(other_player_gets[key], int) for key in self.dond_game.items)):
                        errors.append(f'The "other_player_gets" value must be a dictionary with integer values {self.dond_game.items}.')


                    if not re.search(self.finalize_pattern, response, re.DOTALL):
                        errors.append('Could not pattern match on finalization.')


            except json.JSONDecodeError:
                errors.append("The content within <finalize>...</finalize> is not valid JSON.")

        # Generate error message or return success
        if errors:
            return False, "Errors: " + "; ".join(errors)
        else:
            return True, "Response is valid."


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

        
    def new_round(self):
        """
        Resets round attributes.
        """
        self.is_new_round = True
        self.other_has_finalized = False
        self.first_move = True
    
    def new_game(self):
        """
        Resets the message history of the LLM player.

        Returns:
            list: The message history before resetting.
        """
        self.new_round()
        self.round_nb = 1
        self.is_new_game = True
        self.first_move = True
        self.other_has_finalized = False
        history = self.agent.history
        self.agent.reset_messages()
        return history
    
    def get_history(self):
        """
        Returns the current message history of the LLM player.

        Returns:
            list: The message history.
        """
        return self.agent.history
