import regex as re
import json
import random
import os
from collections import deque


class DondGame:
    def __init__(
        self,
        players,
        mode="coop",
        max_turns=None,
        player_order="deterministic",
        setup="random_read",
        setups_file=None,
        rounds_per_game=10,
        items=None,
        quantities=None,
        role_values=None,
        finalization_visibility=False,
    ):
        """
        Initializes the DoND game.

        Args:
            players (list): List of player names.
            mode (str): The mode of the game, either 'coop' or 'basic'.
            max_turns (int): The maximum number of turns allowed in the game.
            player_order (str): The order of players, either 'deterministic' or 'stochastic'.
            setup (str): The setup type, either 'random_read' or 'manual'.
            setups_file (str): The file containing game setups.
            rounds_per_game (int): The number of rounds per game.
            items (list): The list of items in the game.
            quantities (list): The quantities of items.
            finalization_visibility (bool): Visibility of finalization.
        """
        self.players = players
        self.roles = ["starting_negotiator", "responding_negotiator"]
        self.mode = mode
        self.max_turns = max_turns
        self.player_order = player_order
        self.setup = setup
        self.setups_file = setups_file
        self.rounds_per_game = rounds_per_game
        self.quantities = quantities
        self.finalization_visibility = finalization_visibility

        if self.setup == "random_read":
            self.items = ["books", "hats", "balls"]
            self.settings = []
            with open(self.setups_file) as f:
                lines = f.readlines()
                self.nb_settings = len(lines)
                for i in range(0, self.nb_settings, 2):
                    l = [int(item) for item in lines[i].split()]
                    l2 = [int(item) for item in lines[i + 1].split()]
                    quantities = {
                        key: value for key, value in zip(self.items, [l[0], l[2], l[4]])
                    }
                    role_values = {
                        self.roles[0]: {key: value for key, value in zip(self.items, [l[1], l[3], l[5]])},
                        self.roles[1]: {key: value for key, value in zip(self.items, [l2[1], l2[3], l2[5]])}
                    }
                    self.settings.append((quantities, role_values))
            self.nb_settings = len(self.settings)

        elif self.setup == "manual":
            self.items = items
            self.quantities = {key: value for key, value in zip(self.items, quantities)}
            self.role_values = {
                self.roles[0]: {key: value for key, value in zip(self.items, role_values[0])},
                self.roles[1]: {key: value for key, value in zip(self.items, role_values[1])}
            }

        self.new_game()

    def step(self, output, is_finalization=False) -> bool:
        """
        Advances the game by one step.

        Args:
            output (str | list): The output message or finalization list from the player.
            is_finalization (bool): Indicates if the output is a finalization.

        Returns:
            bool: False if game ended else True.
        """
        self.turn += 1
        self.last_message = output
        round_over = False
        game_over = False

        if self.has_finalized:
            if not is_finalization:
                self.points = {player: 0 for player in self.players}
                self.agreement_reached = False
            else:
                self.finalize(output)
                if self.verify_finalizations_match():
                    self.set_points()
                    self.agreement_reached = True
                else:
                    self.points = {player: 0 for player in self.players}
                    self.agreement_reached = False
            round_over = True

        else:
            if is_finalization:
                self.has_finalized = True
                self.finalize(output)

            if self.turn > self.max_turns:
                round_over = True

        self.role_deque.rotate(-1)
        if round_over: self.new_round()
        if self.round_nb > self.rounds_per_game-1:
            game_over = True
        state = self.get_state()
        return round_over, game_over, state


    def verify_finalizations_match(self):
        """
        Verifies if the finalizations from both players match the total quantities.

        Returns:
            bool: True if the finalizations match, False otherwise.
        """
        for item in self.items:
            total = sum(self.role_props[role][item] for role in self.roles)
            if total != self.quantities[item]:
                return False
        return True

    def set_points(self):
        """
        Sets the points for both players based on their finalizations.
        """
        utilities = {
            role: sum(self.role_values[role][item] * self.role_props[role][item] for item in self.items)
            for role in self.roles
        }

        if self.mode == "coop":
            total = sum(utilities.values())
            self.points = {player: total for player in self.players}


        elif self.mode == "basic":
            self.points = {self.role_to_player[role]: utilities[role] for role in self.roles}

    def finalize(self, finalization: list):
        """
        Records the finalization from the current player.

        Args:
            finalization (list): The list of finalized quantities for each item.
        """
        current_role = self.current_turn()
        self.role_props[current_role] = finalization["i_take"]

    def get_state(self):
        """
        Retrieves the current state of the game.

        Returns:
            dict: The current state of the game.
        """
        out = {
            "mode": self.mode,
            "role_values": self.role_values,
            "role_props": self.role_props,
            "player_to_role": self.player_to_role,
            "is_new_round": True if self.turn <= 1 else False,
            "is_new_game": True if (self.turn <= 1 and self.round_nb == 0) else False,
            "items": self.items,
            "turn": self.turn,
            "current_turn": self.current_turn(),
            "round_number": self.round_nb,
            "nb_rounds": self.rounds_per_game,
            "quantities": self.quantities,
            "has_finalized": self.has_finalized,
            "last_message": self.last_message,
            "finalization_visibility": self.finalization_visibility,
            "round_agreements": self.round_agreement_reached,
            "round_points": {player: self.round_points[player] for player in self.players},
        }
        return out


    def set_new_game_settings(self):
        """
        Sets new game settings based on the setup type.
        """
        if self.setup == "manual":
            return

        elif self.setup == "random_read":
            setting_id = random.randint(0, self.nb_settings - 1)
            self.quantities, self.role_values = self.settings[setting_id]

    def archive_player_states(self):
        """
        Archives the states of the players for the current round.
        """
        for role in self.roles:
            player = self.role_to_player[role]
            self.round_role_props[role].append(self.role_props[role])
            self.round_points[player].append(self.points[player])
            self.round_values[role].append(self.role_values[role])
        self.round_quantities.append(self.quantities)
        self.round_agreement_reached.append(self.agreement_reached)
        self.round_role_to_player.append(self.role_to_player.copy())

    def new_game(self):
        """
        new_games the game to its initial state.

        Returns:
            tuple: The quantities of items and the values for player 0 and player 1.
        """
        self.has_finalized = False
        self.role_props = {role: {} for role in self.roles}
        self.points = {player: 0 for player in self.players}
        self.agreement_reached = False
        self.last_message = None
        self.round_nb = 0
        self.turn = 0
        self.round_ended = False
        self.game_ended = False
        self.last_message = None
        self.round_role_props = {role: [] for role in self.roles}
        self.round_points = {player: [] for player in self.players}
        self.round_values = {role: [] for role in self.roles}
        self.round_quantities = []
        self.round_agreement_reached = []
        self.round_role_to_player = []
        self.role_deque = self.get_new_role_deque()
        self.set_new_game_settings()
        self.assign_roles()

    def new_round(self):
        """
        Ends the current round and prepares for the next round.
        """
        self.archive_player_states()
        self.round_nb += 1
        self.has_finalized = False
        self.role_props = {role: {} for role in self.roles}
        self.points = {player: 0 for player in self.players}
        self.agreement_reached = False
        self.last_message = None
        self.turn = 0
        self.round_ended = True
        self.last_message = None
        self.set_new_game_settings()
        self.role_deque = self.get_new_role_deque()
        self.assign_roles()
    
    def get_new_role_deque(self):
        """
        Set the order of roles.
        """
        if self.player_order == "deterministic":
            return deque(self.roles)
        elif self.player_order == "stochastic":
            return deque(random.sample(self.roles, len(self.roles)))
    
    def get_current_player(self):
        """
        Get the current player (the one who has to play next)
        """
        return self.role_to_player[self.role_deque[0]]

    def current_turn(self):
        """
        Determines the current role's turn.

        Returns:
            str: The name of the current role.
        """
        return self.role_deque[0]

    def assign_roles(self):
        """
        Assigns roles to players for the current round.
        """
        if self.player_order == "deterministic":
            self.role_to_player = {role: player for role, player in zip(self.roles, self.players)}
        elif self.player_order == "stochastic":
            shuffled_players = random.sample(self.players, len(self.players))
            self.role_to_player = {role: player for role, player in zip(self.roles, shuffled_players)}
        
        # Create player_to_role mapping
        self.player_to_role = {player: role for role, player in self.role_to_player.items()}

    def export_game(self, file_path):
        """
        Exports the round history lists to a JSON file at the specified file path.

        Args:
            file_path (str): The path to the file where the JSON data will be saved.
        """
        folder_path = os.path.dirname(file_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        round_history = {
            "round_role_props": self.round_role_props,
            "round_points": self.round_points,
            "round_values": self.round_values,
            "round_quantities": self.round_quantities,
            "round_agreement_reached": self.round_agreement_reached,
            "round_role_to_player": self.round_role_to_player,
        }

        with open(file_path, "w") as f:
            json.dump(round_history, f, indent=4)