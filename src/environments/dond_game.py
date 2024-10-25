import regex as re
import json
import random
import os
from collections import deque
import copy

class DondGame:
    def __init__(
        self,
        players,
        mode="coop",
        max_turns=None,
        rounds_per_game=1,
        player_order="deterministic",
        random_setup_func=None,
        random_setup_kwargs=None,
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
        self.random_setup_func = globals()[random_setup_func]
        self.random_setup_kwargs = random_setup_kwargs
        self.finalization_visibility = finalization_visibility
        self.rounds_per_game = rounds_per_game

        self.new_game()

    def set_new_setup(self):
        """
        # TODO: write config
        """
        self.items, self.quantities, role_values = self.random_setup_func(**self.random_setup_kwargs)
        self.role_values = {
            self.roles[0]: role_values[0],
            self.roles[1]: role_values[1]
        }



    def step(self, action) -> bool:
        """
        Advances the game by one step.

        Args:
            output (str | list): The output message or finalization list from the player.
            is_finalization (bool): Indicates if the output is a finalization.

        Returns:
            bool: False if game ended else True.
        """
        is_finalization, output = action
        self.turn += 1
        self.last_message = output
        self.round_ended = False
        self.is_new_round = True if self.turn <= 1 else False
        self.is_new_game = True if (self.turn <= 1 and self.round_nb == 0) else False
        self.game_over = False
        round_over = False

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
        if round_over: 
            self.new_round()
        if self.round_nb > self.rounds_per_game-1:
            self.game_over = True
        state = self.get_state()
        return state


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
        Sets the points for each role based on their finalizations.
        """
        utilities = {
            role: sum(self.role_values[role][item] * self.role_props[role][item] for item in self.items)
            for role in self.roles
        }

        if self.mode == "coop":
            total = sum(utilities.values())
            self.points = {role: total for role in self.roles}

        elif self.mode == "basic":
            self.points = {role: utilities[role] for role in self.roles}

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
        state = {
            "mode": self.mode,
            "role_values": self.role_values,
            "role_props": self.role_props,
            "player_to_role": self.player_to_role,
            "is_new_round": self.is_new_round,
            "is_new_game": self.is_new_game,
            "game_over": self.game_over,
            "items": self.items,
            "turn": self.turn,
            "max_turns": self.max_turns,
            "current_player": self.get_current_player(),
            "round_number": self.round_nb,
            "nb_rounds": self.rounds_per_game,
            "quantities": self.quantities,
            "has_finalized": self.has_finalized,
            "last_message": self.last_message,
            "finalization_visibility": self.finalization_visibility,
            # rounds history
            "round_player_roles": self.round_player_roles,
            "round_quantities": self.round_quantities,
            "round_values": self.round_values,
            "round_finalizations": self.round_finalizations,
            "round_agreements_reached": self.round_agreements_reached,
            "round_points": self.round_points,
        }
        return state

    def archive_player_states(self):
        """
        Archives the states of the players for the current round.
        """
        # Ensure points are initialized for all roles
        if not all(role in self.points for role in self.roles):
            self.points = {role: 0 for role in self.roles}
        
        self.round_player_roles.append(self.player_to_role.copy())
        self.round_quantities.append(self.quantities)
        self.round_values.append({role: self.role_values[role] for role in self.roles})
        self.round_finalizations.append({role: self.role_props[role] for role in self.roles})
        self.round_agreements_reached.append(self.agreement_reached)
        self.round_points.append({role: self.points[role] for role in self.roles})

    def new_round(self):
        """
        Ends the current round and prepares for the next round.
        """
        self.archive_player_states()
        self.round_nb += 1
        self.has_finalized = False
        self.role_props = {role: {} for role in self.roles}
        self.points = {role: 0 for role in self.roles}  # Ensure points are reset
        self.agreement_reached = False
        self.last_message = None
        self.turn = 0
        self.last_message = None
        self.set_new_setup()
        self.role_deque = self.get_new_role_deque()
        self.assign_roles()


    def new_game(self, checkpoint=None):
        """
        Resets the game to its initial state or to a checkpoint if provided.

        Args:
            checkpoint (dict, optional): A dictionary containing the checkpoint state.
        """
        if checkpoint:
            self.load_checkpoint(checkpoint)
        else:
            self.has_finalized = False
            self.role_props = {role: {} for role in self.roles}
            self.points = {role: 0 for role in self.roles}  # Ensure points are initialized
            self.agreement_reached = False
            self.last_message = None
            self.round_nb = 0
            self.turn = 0
            self.is_new_round = True
            self.is_new_game = True
            self.game_over = False
            self.last_message = None
            self.role_deque = self.get_new_role_deque()
            self.set_new_setup()
            self.assign_roles()
            self.round_player_roles = []  # Initialize round_player_roles
            self.round_quantities = []    # Ensure round_quantities is initialized
            self.round_values = []        # Ensure round_values is initialized
            self.round_finalizations = [] # Ensure round_finalizations is initialized
            self.round_agreements_reached = [] # Ensure round_agreement_reached is initialized
            self.round_points = []        # Ensure round_points is initialized

    def load_checkpoint(self, checkpoint):
        """
        Loads the game state from a checkpoint.

        Args:
            checkpoint (dict): A dictionary containing the checkpoint state.
        """
        self.__dict__.update(checkpoint)

    
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


def uniform_quant_random_vals(items, min_quant, max_quant, min_val, max_val):
    quant = random.randint(min_quant, max_quant)
    val_player_0 = [random.randint(min_val, max_val) for _ in range(quant)]
    val_player_1 = copy.deepcopy(val_player_0)
    random.shuffle(val_player_1)
    val_player_0 = {item: val for item, val in zip(items, val_player_0)}
    val_player_1 = {item: val for item, val in zip(items, val_player_1)}
    quantities = {item:q for item,q in zip(items, [quant]*len(items))}
    return items, quantities, (val_player_0, val_player_1)

def independent_random_vals(items, min_quant, max_quant, min_val, max_val):
    quantities = {item: random.randint(min_quant, max_quant) for item in items}
    val_player_0 = {item: random.randint(min_val, max_val) for item in items}
    val_player_1 = {item: random.randint(min_val, max_val) for item in items}
    return items, quantities, (val_player_0, val_player_1)

def fixed_manual(items, quantities, val_player_0, val_player_1):
    quantities = {item: q for item, q in zip(items, quantities)}
    val_player_0 = {item: v for item, v in zip(items, val_player_0)}
    val_player_1 = {item: v for item, v in zip(items, val_player_1)}
    return items, quantities, (val_player_0, val_player_1)

def random_quant_fixed_vals(items, min_quant, max_quant, val_player_0, val_player_1):
    quantities = {item: random.randint(min_quant, max_quant) for item in items}
    val_player_0 = {item: v for item, v in zip(items, val_player_0)}
    val_player_1 = {item: v for item, v in zip(items, val_player_1)}
    return items, quantities, (val_player_0, val_player_1)




