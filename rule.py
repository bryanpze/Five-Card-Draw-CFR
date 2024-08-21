class Rule:
    def __init__(self, num_players=2, stack=[40,40] ,blinds=[1,2],bet_abstraction=1,card_abstraction = None):
        assert(num_players==len(stack))
        assert(num_players)
        self.bet_abstraction = bet_abstraction
        self.num_players = num_players
        self.stack = stack
        self.blinds = blinds
        self.card_abstraction = card_abstraction

