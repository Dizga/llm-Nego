import gymnasium
import regex as re
import json

class DoND():
    def __init__(self,
            instructions: str):
        # Generate values for each player
        self.instructions = instructions
        self.turn = 0
        self.quantities = {'books': 5, 'hats': 4, 'balls': 3}
        self.values_p0 = {'books': 5, 'hats': 4, 'balls':3}
        self.values_p1 = {'books': 5, 'hats': 4, 'balls':3}
        self.has_proposed = False # whether at least one player has proposed
        self.p0_prop = {}
        self.p1_prop = {}
        self.points_p0 = 0
        self.points_p1 = 0
        self.agreement_reached = False

    def step(self, output: str):
        "Play a move. Returns True if game ongoing, False when over."
        self.turn += 1
        # player has proposal or needs to propose
        if self.has_proposed:
            if self.propose(output) != False and self.verify_props_match():
                self.set_points()
            return -1 # game ended with failure to be complementary
        if self.propose(output):
            self.has_proposed = True
            return -1 # game ended because of wrong proposal
        # no proposal, continue conversation
        if check_if_message(self, output): # check message format
            return output
        return -2 # game ended because of bad formatting

    def verify_props_match(self):
        if self.p0_prop['books']*self.p1_prop['books']!=self.quantities['books']: return False
        if self.p0_prop['hats']*self.p1_prop['hats']!=self.quantities['hats']: return False
        if self.p0_prop['balls']*self.p1_prop['balls']!=self.quantities['balls']: return False

    def set_points(self):
        self.points_p0 = (self.values_p0['books']*self.p0_prop['books'] +
        self.values_p0['hats']*self.p0_prop['hats'] +
        self.values_p0['balls']*self.p0_prop['balls'])
        self.points_p1 =  (self.values_p1['books']*self.p1_prop['books'] +
        self.values_p1['hats']*self.p1_prop['hats'] +
        self.values_p1['balls']*self.p1_prop['balls'])

    def get_description(self, player) -> str:
        "Add instructions to model input string"
        if self.current_turn() == "p0":
            return  f"""
                    There is a total of {self.quantities['books']} books,
                    {self.quantities['hats']} hats and {self.quantities['balls']} balls.
                    You values are {self.values_p0['book']} for a book,
                        {self.values_p0['hat']} for a hat and {self.values_p0['ball']} for a ball.
                """

        return f"""
                There is a total of {self.quantities['books']} books,
                {self.quantities['hats']} hats and {self.quantities['balls']} balls.
                You values are {self.values_p1['book']} for a book,
                    {self.values_p1['hat']} for a hat and {self.values_p1['ball']} for a ball.
            """

    def check_if_message(self, string: str) -> bool:
        # TODO: fix
        return re.search(r"\[ Message \] .*", string)

    def propose(self, string: str) -> bool:
        "Sets proposal."
        # TODO: fix
        if re.search(r"\[ Proposal \] \{'books': \d+, 'hats': \d+, 'balls': \d+\}", string) == False:
            return False
        prop = json.loads(string)
        if prop['books'] > self.quantities['books']: return False
        if prop['hats'] > self.quantities['hats']: return False
        if prop['balls'] > self.quantities['balls']: return False
        if self.current_turn() == "a": self.p0_prop = prop
        else: self.p1_prop = prop
        return True

    def reset(self):
        self.turn = 0
        self.has_proposed = False

    def render(self):
        "Render the interations without chain of thought."

    def export(self):
        "Export the game in Json Format."
        return {
            'p0_score': self.points_p0,
            'p1_score': self.points_p1,
            'quantities': self.quantities,
            'p0_values': self.values_p0,
            'p1_values': self.values_p1,
            'p0_proposal': self.p0_prop,
            'p1_proposal': self.p1_prop,
            'reach_agreement': self.agreement_reached,
        }

    def current_turn(self):
        if self.turn % 2==0: return "a"
        return "b"
