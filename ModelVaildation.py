from MCTS import MCTS, puremcts_policy
from OthelloGame import Game
import numpy as np
import torch
import argparse
from DNN import Network

def convert(act_list):
    s = ""
    for num in act_list:
        if 0 <= num < 64:
            s += chr(ord('A') + num%8)
            s += chr(ord('1') + 7-num//8)
    return s

class Arena():
    def __init__(self, NNET1_PATH, NNET2_PATH):
        self.use_gpu = torch.cuda.is_available() 
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        if NNET1_PATH:
            print(f'load model1 from {NNET1_PATH}')
            checkpoint = torch.load(NNET1_PATH,map_location=self.device)
            self.nnet1 = Network(4, 128, 15, 64, 65).to(self.device)
            self.nnet1.load_state_dict(checkpoint['cur_nnet'])
            self.nnet1.eval()
        else:
            self.nnet1 = puremcts_policy
            print(f'cannot load model1, use pure mcts.')
        if NNET2_PATH:
            print(f'load model2 from {NNET2_PATH}')
            checkpoint = torch.load(NNET2_PATH,map_location=self.device)
            self.nnet2 = Network(4, 128, 15, 64, 65).to(self.device)
            self.nnet2.load_state_dict(checkpoint['cur_nnet'])
            self.nnet2.eval()
        else:
            self.nnet2 = puremcts_policy
            print(f'cannot load model2, use pure mcts.')
    
    def run(self, n_games=20, n_playout=800, c_puct=5, temperature=0.001, show=True):
        mcts1 = MCTS(self.nnet1, n_playout, c_puct)
        mcts2 = MCTS(self.nnet2, n_playout, c_puct)
        score = 0
        cur_nnet_player = 1
        for i in range(n_games):
            print('now nnet1 is',cur_nnet_player)
            game = Game(self.device)
            moves_count = 0
            act_list = []
            #game.print_game_state()
            while game.get_game_result() == 0:
                if game.cur_player == cur_nnet_player:
                    if moves_count < 8:
                        acts, act_probs = mcts1.play(game, 1)
                    else:
                        acts, act_probs = mcts1.play(game, temperature)
                    action = int(np.random.choice(acts, p=act_probs))
                    act_list.append(action)
                    mcts1.update_with_move(-1)
                    game.set_next_state(action)
                    moves_count += 1
                else:
                    if moves_count < 8:
                        acts, act_probs = mcts2.play(game, 1)
                    else:
                        acts, act_probs = mcts2.play(game, temperature)
                    action = int(np.random.choice(acts, p=act_probs))
                    act_list.append(action)
                    mcts2.update_with_move(-1)
                    game.set_next_state(action)
                    moves_count += 1
            game.print_game_state()
            print(f'Game{i+1}: ' + convert(act_list))
            res = game.get_game_result()
            if res == game.Draw:
                res = 0
            score += res * game.cur_player * cur_nnet_player
            if res * game.cur_player * cur_nnet_player > 0:
                print('nnet1 wins!')
            cur_nnet_player = -cur_nnet_player
        return score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1',type=str)
    parser.add_argument('--model2',type=str)
    parser.add_argument('-n',type=int,default=20)
    parser.add_argument('-p',type=int,default=800)
    parser.add_argument('-c',type=int,default=5)
    args = parser.parse_args()
    arena = Arena(args.model1,args.model2)
    score = arena.run(args.n,args.p,args.c)
    print('final score =',score)
