from MCTS import MCTS, puremcts_policy
from OthelloGame import Game
import numpy as np
import torch
import argparse
from DNN import Network

def convert_single(num):
    s = ""
    if 0 <= num < 64:
        s += chr(ord('A') + num%8)
        s += chr(ord('1') + 7-num//8)
    return s    

def iconvert_single(s):
    return ord(s[0])-ord('A') + (7 - ord(s[1]) + ord('1')) * 8 

def convert(act_list):
    s = ""
    for num in act_list:
        s += convert_single(num)
    return s

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',type=str)
    parser.add_argument('--cur',type=int,default=1)
    parser.add_argument('-p',type=int,default=800)
    parser.add_argument('-c',type=int,default=5)
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available() 
    device = torch.device("cuda" if use_gpu else "cpu")

    print('device:',device)

    print(f'load model1 from {args.model}')
    checkpoint = torch.load(args.model,map_location=device)
    nnet1 = Network(4, 128, 15, 64, 65).to(device)
    nnet1.load_state_dict(checkpoint['cur_nnet'])
    nnet1.eval()

    mcts1 = MCTS(nnet1, args.p, args.c)
    cur_nnet_player = -args.cur

    game = Game(device)

    moves_count = 0
    temperature = 0.001

    game.print_game_state_on_colab()
    while game.get_game_result()[0] == 0:
        if game.cur_player == cur_nnet_player:
            if moves_count < 8:
                acts, act_probs = mcts1.play(game, 0.1)
            else:
                acts, act_probs = mcts1.play(game, temperature)
            action = int(np.random.choice(acts, p=act_probs))
            if action == 64: 
                print('nnet pass.')
            else:
                print('nnet:',convert_single(action))
            print('score:',mcts1.get_root_score())
            mcts1.update_with_move(-1)
            game.set_next_state(action)
            moves_count += 1
        else:
            playable = np.where(game.get_vaild_moves() == 1)[0]
            playable.sort()
            if playable[-1] == 64:
                print('player pass.')
                action = 64
            else:
                action = -1
                print('player:',end=' ')
                while action not in playable:
                    ss = [convert_single(num) for num in playable]
                    print(ss)
                    s = input().upper()
                    action = iconvert_single(s)
            game.set_next_state(action)
            moves_count += 1
        game.print_game_state_on_colab()
    res = game.get_game_result()[1]
    if res == game.Draw:
        print('draw')
    elif res * game.cur_player * cur_nnet_player > 0:
        print('nnet wins!')
    else:
        print('player wins!')