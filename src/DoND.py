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
        self.has_proposed = False
        self.p0_prop
        self.b_prop = {}
        self.points_p0 = 0
        self.points_p1 = 0

    def step(self, output: str):
        "Play a move. Returns True if game ongoing, False when over."
        self.turn += 1

        # player has proposal or needs to propose
        if self.has_proposed:
            if self.propose(output) != False and self.verify_props_match():
                self.set_points()
            return False # game ended

        if self.propose(output):
            self.has_proposed = True
            if self.current_turn() == "a":
                self.p1_perspective += output + "<You> : "
            return True

        # no proposal, continue conversation
        if self.current_turn() == "a":
            # TODO: verify
            self.p1_perspective += output + "<You> : "
            self.a_perspective += output + "<Other> : "
            return self.wrap()

        self.a_perspective += output + "<You> : "
        self.p1_perspective += output + "<Other> : "
        return self.wrap()

    def verify_props_match(self):
        if self.p0_prop['books']*self.b_prop['books']!=self.quantities['books']: return False
        if self.p0_prop['hats']*self.b_prop['hats']!=self.quantities['hats']: return False
        if self.p0_prop['balls']*self.b_prop['balls']!=self.quantities['balls']: return False

    def set_points(self):
        self.points_p0 = (self.values_p0['books']*self.p0_prop['books'] +
        self.values_p0['hats']*self.p0_prop['hats'] +
        self.values_p0['balls']*self.p0_prop['balls'])
        self.points_p1 =  (self.values_p1['books']*self.b_prop['books'] +
        self.values_p1['hats']*self.b_prop['hats'] +
        self.values_p1['balls']*self.b_prop['balls'])

    def wrap(self) -> str:
        "Add instructions to model input string"
        if self.current_turn() == "a":
            return self.instructions + f"""<Game Start>
                    There is a total of {self.quantities['books']} books,
                    {self.quantities['hats']} hats and {self.quantities['balls']} balls.
                    You values are {self.values_p0['book']} for a book,
                        {self.values_p0['hat']} for a hat and {self.values_p0['ball']} for a ball.
                """ + self.a_perspective

        return self.instructions + f"""<Game Start>
                There is a total of {self.quantities['books']} books,
                {self.quantities['hats']} hats and {self.quantities['balls']} balls.
                You values are {self.values_p1['book']} for a book,
                    {self.values_p1['hat']} for a hat and {self.values_p1['ball']} for a ball.
            """ + self.p1_perspective

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
        else: self.b_prop = prop
        return True

    def reset(self):
        self.turn = 0
        self.has_proposed = False
        self.a_perspective = "You: "
        self.p1_perspective = ""

    def render(self):
        "Render the interations without chain of thought."

    def export(self):
        "Export the game in Json Format."

    def close(self):
        pass

    def current_turn(self):
        if self.turn % 2==0: return "a"
        return "b"
