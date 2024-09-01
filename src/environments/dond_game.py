import regex as re
import json
import random

class DondGame:
    def __init__(self, 
                 mode='coop',
                 max_turns=None,
                 setup="random_read",
                 setups_file=None,
                 rounds_per_game = 10,
                 items = None,
                 player_0_values = None,
                 player_1_values = None,
                 quantities = None,
                 ):
        """
        Initializes the DoND game.

        Args:
            max_turns (int): The maximum number of turns allowed in the game.
        """
        self.mode = mode
        self.max_turns = max_turns
        self.setup = setup
        self.setups_file = setups_file
        self.rounds_per_game = rounds_per_game


        if self.setup == "random_read":
            self.items = ['books', 'hats', 'balls']
            self.settings = []
            # Get dataset of game setups from file
            with open(self.setups_file) as f:
                lines = f.readlines()
                self.nb_settings = len(lines)
                for i in range(0, self.nb_settings, 2):
                    # TODO: ensure that quantities match!
                    l = [int(item) for item in lines[i].split()]
                    l2 = [int(item) for item in lines[i+1].split()]
                    quantities = {key: value for key, value in zip(self.items, [l[0], l[2], l[4]])}
                    player_0_values = {key: value for key, value in zip(self.items, [l[1], l[3], l[5]])}
                    player_1_values = {key: value for key, value in zip(self.items, [l2[1], l2[3], l2[5]])}
                    self.settings.append((quantities, player_0_values, player_1_values))
            self.nb_settings = len(self.settings)
            
        
        elif self.setup == "manual":
            self.items = items
            self.quantities = {key: value for key, value in zip(self.items, quantities)}
            self.values_player_0 = {key: value for key, value in zip(self.items, player_0_values)}
            self.values_player_1 = {key: value for key, value in zip(self.items, player_1_values)}

        self.reset()


    def reset(self):
        """
        Resets the game to its initial state.

        Returns:
            tuple: The quantities of items and the values for player 0 and player 1.
        """
        self.turn = 0
        self.has_finalized = False

        # changes the quantities and values of the players
        self.set_new_game_settings()

        self.reset_player_bookkeeping()
        self.round_nb = 1
        self.new_round = True
        self.game_ended = False
        self.player_0_prop_history = []
        self.player_1_prop_history = []
        self.points_player_0_history = []
        self.points_player_1_history = []
        self.values_player_0_history = []
        self.values_player_1_history = []
        self.quantities_history = []
        self.agreement_reached_history = []

        return self.get_state()
    
    def set_new_game_settings(self):

        if self.setup == "manual":
            return

        # Pick random trio of quantities & values from dataset
        elif self.setup == "random_read":
            setting_id = random.randint(0, self.nb_settings-1)
            self.quantities, self.values_player_0, self.values_player_1 = self.settings[setting_id]

    
    def reset_player_bookkeeping(self):
        self.player_0_prop = {}
        self.player_1_prop = {}
        self.points_player_0 = 0
        self.points_player_1 = 0
        self.agreement_reached = False
        self.last_message = ""

    def archive_player_states(self):
        self.player_0_prop_history.append(self.player_0_prop)
        self.player_1_prop_history.append(self.player_1_prop)
        self.points_player_0_history.append(self.points_player_0)
        self.points_player_1_history.append(self.points_player_1)
        self.values_player_0_history.append(self.values_player_0)
        self.values_player_1_history.append(self.values_player_1)
        self.quantities_history.append(self.quantities)
        self.agreement_reached_history.append(self.agreement_reached)

    def end_round(self):
        self.round_nb += 1
        self.turn = 0
        self.has_finalized = False
        self.archive_player_states()
        self.reset_player_bookkeeping()
        self.new_round = True
        if self.round_nb > self.rounds_per_game:
            self.game_ended = True
        self.set_new_game_settings()

    
    def step(self, output, is_finalization=False)-> bool: 
        """
        Advances the game by one step.

        Args:
            output (str | list): The output message or finalization list from the player.
            is_finalization (bool): Indicates if the output is a finalization.

        Returns:
            bool: False if game ended else True.
        """
        
        self.last_message = output

        if self.has_finalized:
            if not is_finalization: # player has not made a finalization after other player, automatic loss
                self.end_round()
                return self.get_state()
            self.finalize(output)
            if self.verify_props_match():
                self.set_points()
                self.agreement_reached = True
            self.end_round()
            return self.get_state()  

        self.has_finalized = is_finalization

        if is_finalization:
            self.has_finalized = True
            self.finalize(output)
        
        self.turn += 1

        if self.turn > self.max_turns:
            self.end_round()
            return self.get_state()  # round ended due to exceeding max turns
        
        self.new_round = False

        return self.get_state()  # game not ended

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
            'game_ended': self.game_ended,
            "new_round": self.new_round,
            "current_turn": self.current_turn(),
            "round_number": self.round_nb,
            "quantities": self.quantities,
            "agreement_reached": self.agreement_reached,
            "has_finalized": self.has_finalized,
            "last_message": self.last_message
        }
        if player == "player_0":
            out["values"] = self.values_player_0
            out["last_score"] = self.points_player_0
            return out

        out["values"] = self.values_player_1
        out["last_score"] = self.points_player_1
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
        points_player_0 = sum(self.values_player_0[item] * self.player_0_prop[item] for item in self.items)
        points_player_1 = sum(self.values_player_1[item] * self.player_1_prop[item] for item in self.items)

        if self.mode == "coop":
            total = points_player_0 + points_player_1
            self.points_player_0 = total
            self.points_player_1 = total

        elif self.mode == "basic":
            self.points_player_0 = points_player_0
            self.points_player_1 = points_player_1

    def finalize(self, finalization: list):
        """
        Records the finalization from the current player.

        Args:
            finalization (list): The list of finalized quantities for each item.
        """
        if self.current_turn() == "player_0":
            self.player_0_prop = finalization['i_take']
        else:
            self.player_1_prop = finalization['i_take']

    def render(self):
        """
        Renders the current state of the game (not implemented).
        """
        pass

    def export(self):
        """
        Export round metrics.
        """
        rounds = []
        for round_id in range(self.rounds_per_game):
            rounds.append(self.export_round(round_id))
        return rounds

    def export_summary(self):
        return {
            'player_0_total_reward': sum(self.points_player_0_history),
            'player_1_total_reward': sum(self.points_player_1_history),
        }

    def export_round(self, id=-1):
        """
        Exports the current state of the game as a dictionary.

        Returns:
            dict: The current state of the game.
        """
        return {
            'mode': self.mode,
            'round_id': id,
            'player_0_reward': self.points_player_0_history[id],
            'player_1_reward': self.points_player_1_history[id],
            'quantities': self.quantities_history[id],
            'player_0_values': self.values_player_0_history[id],
            'player_1_values': self.values_player_1_history[id],
            'player_0_finalization': self.player_0_prop_history[id],
            'player_1_finalization': self.player_1_prop_history[id],
            'agreement_reached': self.agreement_reached_history[id],
        }

    def current_turn(self):
        """
        Determines the current player's turn.

        Returns:
            str: 'player_0' if it's player 0's turn, 'player_1' if it's player 1's turn.
        """
        return "player_0" if self.turn % 2 == 0 else "player_1"
