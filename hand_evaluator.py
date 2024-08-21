import math
import functools
import struct

class Evaluator:
    def __init__(self):
        self.card_vals = ['2', '3', '4', '5', '6', '7', '8', '9', 't', 'j', 'q', 'k', 'a'],
        self.cards = {
        "2c": 1,
        "2d": 2,
        "2h": 3,
        "2s": 4,
        "3c": 5,
        "3d": 6,
        "3h": 7,
        "3s": 8,
        "4c": 9,
        "4d": 10,
        "4h": 11,
        "4s": 12,
        "5c": 13,
        "5d": 14,
        "5h": 15,
        "5s": 16,
        "6c": 17,
        "6d": 18,
        "6h": 19,
        "6s": 20,
        "7c": 21,
        "7d": 22,
        "7h": 23,
        "7s": 24,
        "8c": 25,
        "8d": 26,
        "8h": 27,
        "8s": 28,
        "9c": 29,
        "9d": 30,
        "9h": 31,
        "9s": 32,
        "tc": 33,
        "td": 34,
        "th": 35,
        "ts": 36,
        "jc": 37,
        "jd": 38,
        "jh": 39,
        "js": 40,
        "qc": 41,
        "qd": 42,
        "qh": 43,
        "qs": 44,
        "kc": 45,
        "kd": 46,
        "kh": 47,
        "ks": 48,
        "ac": 49,
        "ad": 50,
        "ah": 51,
        "as": 52
        }
        self.handtypes = [
        "invalid hand",
        "high card",
        "one pair",
        "two pairs",
        "three of a kind",
        "straight",
        "flush",
        "full house",
        "four of a kind",
        "straight flush"
        ]
        with open('./HandRanks.dat','rb') as file:
            self.ranks_data = file.read()
    def fill_hand(self,cards):
        cards_used =  [0,0,0,0,0,0,0,0,0,0,0,0,0]
        for card in cards:
            i = math.floor((self.cards[card.lower()]-1)/4)
            cards_used[i] = 1
        to_fill = 2
        for i in range(13):
            if to_fill==0:
                break
            if cards_used[i]==0 and not self.make_straight(cards_used,i):
                cards_used[i]==2
                to_fill-=1
        suit = ['s','d']
        for i in range(14):
            if cards_used[i]==2:
                card = self.card_vals[i]+suit[0]
                suit.pop(0)
                cards.append(card)
        return cards
    
    @staticmethod
    def reducing_function(prev,next):
        if prev==5:
            return 5
        else:
            return prev+1 if next else 0
    def make_straight(self,cards_used,rank):
        new_cards = [cards_used[12]]+cards_used
        new_cards[rank+1]=2
        tmp = functools.reduce(self.reducing_function,new_cards)
        return 5==tmp
    def eval_hand(self,cards):
        if len(cards)==3:
            cards =  self.fill_hand(cards)
        if type(cards[0])==str:
            cards = list(map(lambda card: self.cards[card.lower()],cards))
        return self.evaluate(cards)
    
    def evaluate(self,cards):
        p=53
        for i in range(len(cards)):
            p=self.eval_card(p+cards[i])
        if len(cards)==5 or len(cards)==6:
            p = self.eval_card(p)
        return {
            'handType': p >> 12,
            'handRank': p & 0x00000fff,
            'value': p,
            'handName': self.handtypes[p >> 12]
        }
    def eval_card(self,card):
        offset = card * 4
        return struct.unpack('<I', self.ranks_data[offset:offset+4])[0] 