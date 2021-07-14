import torch
from OthelloGame import Game
import numpy as np
import copy
import logging
import DNN

log = logging.getLogger(__name__)

# Reference from:
# https://github.com/junxiaosong/AlphaZero_Gomoku/blob/master/mcts_alphaZero.py
"""
MIT License

Copyright (c) 2017 junxiaosong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class MCTSNode(object):
    """
    A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {} # a map from action to TreeNode
        self._Nsa = 0
        self._Qsa = 0
        self._Usa = 0
        self._Psa = prior_p

    def select(self, c_puct):
        return max(self._children.items(), key=lambda act_node: act_node[1].get_puct(c_puct))

    def get_puct(self, c_puct):
        self._Usa = c_puct * self._Psa * np.sqrt(self._parent._Nsa) / (1 + self._Nsa)
        return self._Qsa + self._Usa

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = MCTSNode(self, prob)

    def backup(self, leaf_value):
        if self._parent:
            self._parent.backup(-leaf_value)
        self._Nsa += 1
        self._Qsa += (leaf_value - self._Qsa) / self._Nsa

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None

class MCTS():
    def __init__(self, nnet, n_playout, c_puct):
        self.nnet = nnet #[4,8,8] -> [65(moves) + 1(score)]
        self.n_playout = n_playout
        self.c_puct = c_puct
        self._root = MCTSNode(None, 1.0)

    def search(self, game):
        node = self._root
        while(not node.is_leaf()):
            action, node = node.select(self.c_puct)
            game.set_next_state(action)

        result = game.get_game_result()
        if result == 0:
            log_actprob, score = self.nnet(game.get_nnet_format())
            pridiction = np.exp(log_actprob[0].cpu().detach().numpy())
            vailds = game.get_vaild_moves()
            pridiction = pridiction * vailds # masking invaild moves

            sum_action_prob = sum(pridiction)
            if sum_action_prob > 0:
                pridiction /= sum_action_prob
            else:
                log.error("warning: All valid moves were masked.")
                pridiction = vailds
                pridiction /= np.sum(pridiction)

            action = np.nonzero(pridiction)[0]
            prob = pridiction[action]
            node.backup(-float(score.cpu().detach()))
            node.expand(list(zip(action, prob)))
        else: 
            node.backup(-result)

    def play(self, game, temperature):
        for n in range(self.n_playout):
            self.search(copy.deepcopy(game))
        
        act_visits = [(act, node._Nsa) for act, node in self._root._children.items()]

        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temperature * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = MCTSNode(None, 1.0)
    
    def get_root_score(self):
        return self._root._Qsa

def puremcts_policy(input):
    return torch.from_numpy(np.ones(65)), torch.from_numpy(np.zeros(1))

if __name__ == "__main__":
    mcts = MCTS(puremcts_policy,c_puct=5,n_playout=800)
    mcts.nnet
    game = Game(torch.device('cpu'))
    
    game.print_game_state()
    while game.get_game_result() == 0:
        if game.cur_player == -1:
            acts, act_probs = mcts.play(game, 1)
            print(acts)
            print(act_probs)
            action = int(np.random.choice(acts, p=act_probs))
            print('mcts: action ',action)
            mcts.update_with_move(-1)
            game.set_next_state(action)
            game.print_game_state()
        else:
            acts = np.nonzero(game.get_vaild_moves())[0]
            print(acts)
            action = int(np.random.choice(acts))
            print('random:',action)
            mcts.update_with_move(-1)
            game.set_next_state(action)
            game.print_game_state()
    print('winner: ',game.get_game_result() * game.cur_player)
