import regex as re
import json

class DoND:
    def __init__(self, max_turns=7):
        """
        Initializes the DoND game.

        Args:
            max_turns (int): The maximum number of turns allowed in the game.
        """
        self.reset(max_turns)

    def reset(self, max_turns=2):
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
        self.values_p0 = {key: value for key, value in zip(self.items, [5, 4, 3])}
        self.values_p1 = {key: value for key, value in zip(self.items, [5, 4, 3])}
        self.has_proposed = False
        self.p0_prop = {}
        self.p1_prop = {}
        self.points_p0 = 0
        self.points_p1 = 0
        self.agreement_reached = False
        self.last_message = ""
        self.max_turns = max_turns
        return self.quantities, self.values_p0, self.values_p1

    def step(self, output: str | list, is_proposal=False):
        """
        Advances the game by one step.

        Args:
            output (str | list): The output message or proposal list from the player.
            is_proposal (bool): Indicates if the output is a proposal.

        Returns:
            bool: Whether the game should continue or not.
        """
        self.turn += 1
        self.last_message = output
        
        if self.has_proposed:
            self.propose(output)

            if self.verify_props_match():
                self.set_points()
                self.agreement_reached = True
                return False  # Game ended successfully
            return False  # Game ended with failure to be complementary
        
        self.has_proposed = is_proposal
        
        if is_proposal:
            self.has_proposed = True
            self.propose(output)
            return True  # Continue the game
        
        if self.turn > self.max_turns:
            return False  # Game ended due to exceeding max turns
        
        return True  # Continue the game
    
    def get_state(self, player="current_turn"):
        """
        Retrieves the current state of the game.

        Args:
            player (str): The player whose state is to be retrieved ('p0', 'p1', or 'current_turn').

        Returns:
            dict: The current state of the game.
        """
        if player == "current_turn":
            player = self.current_turn()
        out = {
            "quantities": self.quantities,
            "agreement_reached": self.agreement_reached,
            "has_proposed": self.has_proposed,
            "last_message": self.last_message
        }
        if player == "p0":
            out["values"] = self.values_p0
            return out
        out["values"] = self.values_p1
        return out

    def verify_props_match(self):
        """
        Verifies if the proposals from both players match the total quantities.

        Returns:
            bool: True if the proposals match, False otherwise.
        """
        for item in self.items:
            if self.p0_prop[item] + self.p1_prop[item] != self.quantities[item]:
                return False
        return True

    def set_points(self):
        """
        Sets the points for both players based on their proposals.
        """
        self.points_p0 = sum(self.values_p0[item] * self.p0_prop[item] for item in self.items)
        self.points_p1 = sum(self.values_p1[item] * self.p1_prop[item] for item in self.items)

    def propose(self, proposal: list):
        """
        Records the proposal from the current player.

        Args:
            proposal (list): The list of proposed quantities for each item.
        """
        if self.current_turn() == "p0":
            self.p0_prop = {key: value for key, value in zip(self.items, proposal)}
        else:
            self.p1_prop = {key: value for key, value in zip(self.items, proposal)}

    def render(self):
        """
        Renders the current state of the game (not implemented).
        """
        pass

    def export_game(self):
        """
        Exports the current state of the game as a dictionary.

        Returns:
            dict: The current state of the game.
        """
        return {
            'p0_score': self.points_p0,
            'p1_score': self.points_p1,
            'quantities': self.quantities,
            'p0_values': self.values_p0,
            'p1_values': self.values_p1,
            'p0_proposal': self.p0_prop,
            'p1_proposal': self.p1_prop,
            'agreement_reached': self.agreement_reached,
        }

    def current_turn(self):
        """
        Determines the current player's turn.

        Returns:
            str: 'p0' if it's player 0's turn, 'p1' if it's player 1's turn.
        """
        return "p0" if self.turn % 2 == 1 else "p1"
