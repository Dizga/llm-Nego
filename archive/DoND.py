import regex as re
import json

class DoND:
    def __init__(self, max_turns=None):
        """
        Initializes the DoND game.

        Args:
            max_turns (int): The maximum number of turns allowed in the game.
        """
        self.max_turns = max_turns
        self.reset()

    def reset(self):
        """
        Resets the game to its initial state.

        Args:
            max_turns (int): The maximum number of turns allowed in the game.

        Returns:
            tuple: The quantities of items and the values for player 0 and player 1.
        """
        self.turn = 0
        self.items = ['books', 'hats', 'balls']
        self.quantities = {key: value for key, value in zip(self.items, [5, 4, 3])}
        self.values_player_0 = {key: value for key, value in zip(self.items, [5, 4, 3])}
        self.values_player_1 = {key: value for key, value in zip(self.items, [3, 4, 5])}
        self.has_finalized = False
        self.player_0_prop = {}
        self.player_1_prop = {}
        self.points_player_0 = 0
        self.points_player_1 = 0
        self.agreement_reached = False
        self.last_message = ""
        return self.quantities, self.values_player_0, self.values_player_1

    def step(self, output: str | list, is_finalization=False):
        """
        Advances the game by one step.

        Args:
            output (str | list): The output message or finalization list from the player.
            is_finalization (bool): Indicates if the output is a finalization.

        Returns:
            bool: Whether the game should continue or not.
        """
        self.turn += 1
        self.last_message = output
        
        if self.has_finalized:
            if not is_finalization: return False # player has not made a finalization after other player, automatic loss
            self.finalize(output)

            if self.verify_props_match():
                self.set_points()
                self.agreement_reached = True
                return False  # Game ended successfully
            return False  # Game ended with failure to be complementary
        
        self.has_finalized = is_finalization
        
        if is_finalization:
            self.has_finalized = True
            self.finalize(output)
            return True  # Continue the game
        
        if self.turn > self.max_turns:
            return False  # Game ended due to exceeding max turns
        
        return True  # Continue the game
    
    def get_state(self, player="current_turn"):
        """
        Retrieves the current state of the game.

        Args:
            player (str): The player whose state is to be retrieved ('player_0', 'player_1', or 'current_turn').

        Returns:
            dict: The current state of the game.
        """
        if player == "current_turn":
            player = self.current_turn()
        out = {
            "book_cnt": self.quantities["books"],
            "hat_cnt": self.quantities["hats"],
            "ball_cnt": self.quantities["balls"],
            # "quantities": self.quantities,
            "agreement_reached": self.agreement_reached,
            "has_finalized": self.has_finalized,
            "last_message": self.last_message
        }
        if player == "player_0":
            out["book_val"] = self.values_player_0["books"]
            out["hat_val"] = self.values_player_0["hats"]
            out["ball_val"] = self.values_player_0["balls"]

            return out
        out["book_val"] = self.values_player_1["books"]
        out["hat_val"] = self.values_player_1["hats"]
        out["ball_val"] = self.values_player_1["balls"]
        return out

    def verify_props_match(self):
        """
        Verifies if the finalizations from both players match the total quantities.

        Returns:
            bool: True if the finalizations match, False otherwise.
        """
        for item in self.items:
            if self.player_0_prop[item] + self.player_1_prop[item] != self.quantities[item]:
                return False
        return True

    def set_points(self):
        """
        Sets the points for both players based on their finalizations.
        """
        self.points_player_0 = sum(self.values_player_0[item] * self.player_0_prop[item] for item in self.items)
        self.points_player_1 = sum(self.values_player_1[item] * self.player_1_prop[item] for item in self.items)

    def finalize(self, finalization: list):
        """
        Records the finalization from the current player.

        Args:
            finalization (list): The list of finalized quantities for each item.
        """
        if self.current_turn() == "player_0":
            self.player_0_prop = {key: value for key, value in zip(self.items, finalization)}
        else:
            self.player_1_prop = {key: value for key, value in zip(self.items, finalization)}

    def render(self):
        """
        Renders the current state of the game (not implemented).
        """
        pass

    def export(self):
        """
        Exports the current state of the game as a dictionary.

        Returns:
            dict: The current state of the game.
        """
        return {
            'player_0_reward': self.points_player_0,
            'player_1_reward': self.points_player_1,
            'quantities': self.quantities,
            'player_0_values': self.values_player_0,
            'player_1_values': self.values_player_1,
            'player_0_finalization': self.player_0_prop,
            'player_1_finalization': self.player_1_prop,
            'agreement_reached': self.agreement_reached,
        }

    def current_turn(self):
        """
        Determines the current player's turn.

        Returns:
            str: 'player_0' if it's player 0's turn, 'player_1' if it's player 1's turn.
        """
        return "player_0" if self.turn % 2 == 1 else "player_1"
