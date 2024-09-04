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
        self.first_turn = True
        self.is_new_game = True
        
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

    def play_move(self, state):
        """
        Plays a move in the DoND game.

        Returns:
            bool: False if game ended else True.
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
        while retries < self.max_retries:
            if valid_response:
                break
            response = self.agent.prompt(error_message, is_error=True)
            valid_response, error_message = self.validate(response)
            retries += 1

        # Return dummy message if model refuses to conform to correct format
        if retries == self.max_retries:
            response = "<message></message>"

        self.first_turn = False
        self.is_new_game = False
        # Process the response
        return self.extract(response)

    def verificator(self, message):
        """
        Verifies if the message is correct.

        Args:
            message (str): The message to verify.

        Returns:
            bool: Always returns True (placeholder for future implementation).
        """
        # TODO: add conditions that return false if message not correct
        return True

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

        if self.is_new_round: 
            user_message += self.new_round_prompt.format(**state)

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
                    i_take = finalize_json["i_take"]
                    other_player_gets = finalize_json["other_player_gets"]
                    # Generalized validation for arbitrary items
                    if not (isinstance(i_take, dict) and 
                            all(key in i_take for key in self.dond_game.items) and 
                            all(isinstance(i_take[key], int) for key in self.dond_game.items)):
                        errors.append('The "i_take" value must be a dictionary with integer values for the game items.')
                    
                    if not (isinstance(other_player_gets, dict) and 
                            all(key in other_player_gets for key in self.dond_game.items) and 
                            all(isinstance(other_player_gets[key], int) for key in self.dond_game.items)):
                        errors.append('The "other_player_gets" value must be a dictionary with integer values for the game items.')
            except json.JSONDecodeError:
                errors.append("The content within <finalize>...</finalize> is not valid JSON.")

        # Generate error message or return success
        if errors:
            return False, "Errors: " + "; ".join(errors)
        else:
            return True, "Response is valid."

    def extract(self, message):
        """
        Extracts the content from the response message.

        Args:
            message (str): The response message.

        Returns:
            tuple: A tuple containing a boolean indicating if it's a finalization and the extracted content.
        """
        pattern = r'<message>(.*?)</message>|<finalize>\s*\{\s*"i_take"\s*:\s*(.*?)\s*,\s*"other_player_gets"\s*:\s*(.*?)\s*\}\s*</finalize>'
        match = re.search(pattern, message, re.DOTALL)

        if not match:
            return False, ""
        
        elif match.group(2):
            # Extract json from finalization
            i_take = json.loads(match.group(2))
            other_player_gets = json.loads(match.group(3))
            return True, {"i_take": i_take, "other_player_gets": other_player_gets}
        
        else:
            return False, match.group(1)
        
    def new_round(self):
        """
        Resets round attributes.
        """
        self.is_new_round = True
        self.other_has_finalized = False
        self.first_turn = True
    
    def new_game(self):
        """
        Resets the message history of the LLM player.

        Returns:
            list: The message history before resetting.
        """
        self.new_round()
        self.round_nb = 1
        self.is_new_game = True
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
