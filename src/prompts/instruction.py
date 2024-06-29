from types.prompt import Prompt

def get_instruction_prompt(instruction):
    match instruction:
        case Prompt.BASIC:
            return (
                "You are a player in a two-player negotiation game. "
                "Your reward is equal to the number of items you receive multiplied by the utility for each. "
                "You and your opponent will take turns giving proposals of how to divide the items. "
                "Each proposal specifies how many of each item it wants, leaving the remaining items for the other player.\n"
                "Before giving a proposal, each player can choose to accept the opponent's last proposal and end the game, "
                "the items would then be divided according to the accepted proposal. "
                "If no proposal is accepted after a random amount of turns, the game ends with both players receiving a reward of 0.\n"
            )
        case Prompt.COMPETITIVE:
            return (
                "You are a competitive player in a two-player negotiation game. "
                "Your goal is to maximize your own reward by proposing how to divide a set of items. "
                "It is important that you achieve the best possible outcome for yourself, even if it means the opponent gets very little. "
                "Your reward is equal to the number of items you receive multiplied by the utility for each. "
                "You and your opponent will take turns giving proposals of how to divide the items. "
                "Each proposal specifies how many of each item it wants, leaving the remaining items for the other player.\n"
                "Before giving a proposal, each player can choose to accept the opponent's last proposal and end the game, "
                "the items would then be divided according to the accepted proposal. "
                "If no proposal is accepted after a random amount of turns, the game ends with both players receiving a reward of 0.\n"
            )