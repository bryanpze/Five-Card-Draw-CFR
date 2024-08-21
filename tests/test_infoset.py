import sys
import os
sys.path.insert(0,os.path.realpath('.'))
from five_card import PureCFR
from rule import Rule

print('testing infosets')
trainer = PureCFR()
# print(len(trainer.draw_actions_lookup[0]))
trainer.parse_history('ccc')
# assert(trainer.parse_infoset('5s6c8dqstc:cc')['potsize']==4)
# assert(trainer.parse_infoset('5s6c8dqstc:cc')['potsize']==4)
# assert(trainer.parse_infoset('5s6c8dqstc:c'))
#potsize current_player folded_players round_ended
# trainer.parse_infoset('5s6c8dqstc:r10')
# trainer.parse_infoset('5s6c8dqstc:r10c')
# trainer = PureCFR(Rule(num_players=3, stack=[150,150,150]))
# trainer.parse_infoset('5s6c8dqstc:r20r30r40cc/d2d3b3/r12fr21c')
# trainer = PureCFR(Rule(num_players=4, stack=[150,150,150,150]))
# trainer = PureCFR(Rule(num_players=5, stack=[150,150,150,150,150]))