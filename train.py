from os import stat
import time
import datetime
from MCTS import MCTS
from OthelloGame import Game
from DNN import Network
import numpy as np
from collections import deque
import torch
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import random
import argparse
import copy

def convert(act_list):
    s = ""
    for num in act_list:
        if 0 <= num < 64:
            s += chr(ord('A') + num%8)
            s += chr(ord('1') + 7-num//8)
    return s

class TrainPipeline():
    def __init__(self):
        self.use_gpu = torch.cuda.is_available() 
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.epochs = 5
        self.minibatch_steps = 0
        self.batch_size = 512
        self.learn_rate = 1e-2
        self.lr_multiplier = 1.0
        self.momentum = 0.9
        self.L2penalty = 1e-4
        self.epslion = 0.25
        self.eta = 0.3

        self.temperature = 0.001
        self.n_playout = 1600
        self.c_puct = 5

        self.buffer_size = 10000
        self.data_buffer = deque(maxlen=self.buffer_size)
        
        self.eval_freq = 100
        self.check_freq = 20

        self.cur_nnet = Network(4, 128, 15, 64, 65).to(self.device)
        self.best_nnet = Network(4, 128, 15, 64, 65).to(self.device)
        self.optimizer = optim.SGD(self.cur_nnet.parameters(), lr = self.learn_rate, momentum=self.momentum, weight_decay=self.L2penalty)
    
    def add_self_play_data(self, play_data):
        extend_data = []
        for state, mcts_prob, score in play_data:
            for i in [1,2,3,4]:
                rot_state = np.rot90(state, i, (1,2))
                rot_mcts_prob = np.append(np.rot90(mcts_prob[:-1].reshape((8,8)),i).reshape(-1), mcts_prob[-1])
                extend_data.append((rot_state,rot_mcts_prob,score))
                flp_state = np.flip(rot_state, 1)
                flp_mcts_prob = np.append(np.flip(rot_mcts_prob[:-1].reshape((8,8)),1).reshape(-1), rot_mcts_prob[-1])
                extend_data.append((flp_state,flp_mcts_prob,score))
                #print('state\n',state,sep='')
                #print('rot_state\n',rot_state,sep='')
                #print('flp_state\n',flp_state,sep='')
        self.data_buffer.extend(extend_data)        
        #print('debug in add self play:', extend_data[0][0], extend_data[0][1], extend_data[0][2])

    def self_play(self):
        self.best_nnet.eval()
        mcts = MCTS(self.best_nnet, self.n_playout, self.c_puct)
        states, mcts_probs, scores = [],[],[]
        moves_count = 0
        game = Game(self.device)
        while game.get_game_result() == 0:
            # game.print_game_state()
            # For the first 8 moves of each game, the
            # temperature is set to τ = 1; this selects moves proportionally to their visit count in MCTS, and
            # ensures a diverse set of positions are encountered. For the remainder of the game, an infinitesimal
            # temperature is used, τ → 0. 
            if moves_count < 8:
                acts, act_probs = mcts.play(game, 1)
            else:
                acts, act_probs = mcts.play(game, self.temperature)

            # Additional exploration is achieved by adding Dirichlet noise to the prior probabilities in the root node.
            # This noise ensures that all moves may be tried, but the search may still overrule bad moves.
            action = int(np.random.choice(
                acts, 
                p=act_probs * (1-self.epslion) + self.epslion*np.random.dirichlet(self.eta*np.ones(act_probs.shape))
            ))
            mcts.update_with_move(-1)
            
            mcts_prob = np.zeros((65))
            mcts_prob.flat[np.asarray(acts)] = act_probs
            
            states.append(game.get_numpy_format())
            mcts_probs.append(mcts_prob)
            scores.append(game.cur_player)
            #game.print_game_state()
            game.set_next_state(action)
            moves_count += 1
        #game.print_game_state()
        #game.print_game_state()
        
        res = game.get_game_result()
        for i in range(len(scores)):
            if res == game.Draw:
                scores[i] = game.Draw
            else:
                scores[i] = res * game.cur_player * scores[i]
        
        return zip(states, mcts_probs, scores)
    
    def update_nnet(self):
        if len(self.data_buffer) < self.batch_size:
            return
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_prob_batch = [data[1] for data in mini_batch]
        score_batch = [data[2] for data in mini_batch]

        state_batch = Variable(torch.FloatTensor(state_batch)).to(self.device)
        mcts_prob_batch = Variable(torch.FloatTensor(mcts_prob_batch)).to(self.device)
        score_batch = Variable(torch.FloatTensor(score_batch)).to(self.device)

        self.cur_nnet.train()
        self.optimizer.zero_grad()
        self.minibatch_steps += 1
        if self.minibatch_steps % 1000 == 0:
            ts = self.minibatch_steps // 1000
            if 0 < ts <= 400:
                self.lr_multiplier = 1
            elif 400 < ts <= 600:
                self.lr_multiplier = 0.1
            else:
                self.lr_multiplier = 0.01
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learn_rate * self.lr_multiplier
        
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        log_actprobs, score = self.cur_nnet(state_batch)
        #print('debug: ',score.shape, score_batch.shape)
        value_loss = F.mse_loss(score, score_batch.reshape(-1,1))
        policy_loss = -torch.mean(torch.sum(mcts_prob_batch*log_actprobs, 1))
        loss = value_loss + policy_loss

        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()

    def eval_nnet(self, n_games=20):
        mcts1 = MCTS(self.cur_nnet, self.n_playout, self.c_puct)
        mcts2 = MCTS(self.best_nnet, self.n_playout, self.c_puct)
        self.cur_nnet.eval()
        self.best_nnet.eval()
        score = 0
        cur_nnet_player = 1
        for i in range(n_games):
            game = Game(self.device)
            moves_count = 0
            act_list = []
            while game.get_game_result() == 0:
                if game.cur_player == cur_nnet_player:
                    if moves_count < 8:
                        acts, act_probs = mcts1.play(game, 1)
                    else:
                        acts, act_probs = mcts1.play(game, self.temperature)
                    action = int(np.random.choice(acts, p=act_probs))
                    act_list.append(action)
                    mcts1.update_with_move(-1)
                    game.set_next_state(action)
                    moves_count += 1
                else:
                    if moves_count < 8:
                        acts, act_probs = mcts2.play(game, 1)
                    else:
                        acts, act_probs = mcts2.play(game, self.temperature)
                    action = int(np.random.choice(acts, p=act_probs))
                    act_list.append(action)
                    mcts2.update_with_move(-1)
                    game.set_next_state(action)
                    moves_count += 1
            print(f'Game{i+1}: ' + convert(act_list))        
            res = game.get_game_result()
            if res == game.Draw:
                res = 0
            score += res * game.cur_player * cur_nnet_player
            if res * game.cur_player * cur_nnet_player > 0:
                print('nnet1 wins!')
            cur_nnet_player = -cur_nnet_player

        if score >= 1:
            print('cur_nnet become the new best_model')
            self.best_nnet.load_state_dict(self.cur_nnet.state_dict())
            #for param, best_param in zip(self.cur_nnet.parameters(), self.best_nnet.parameters()):
            #    best_param.data.copy_(param.data)
        return score

    def run(self, num_ep):
        print("device:",self.device)
        print('start form', self.minibatch_steps)
        for i in range(self.minibatch_steps+1,self.minibatch_steps+num_ep+1):
            print(f"epoch{i} start at",datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
            print("generating self play data...")
            self.add_self_play_data(self.self_play())
            print("update nnet...")
            loss = self.update_nnet()
            print(f'nnet loss={loss}')
            if (i) % self.check_freq == 0:
                print(f'save checkpoint{self.minibatch_steps+1}...')
                self.save(f'./models/checkpoint{self.minibatch_steps+1}.pt')
            if (i) % self.eval_freq == 0:
                score = self.eval_nnet(20)
                print(f'get score={score}')
                print(f'save checkpoint{self.minibatch_steps+1}...')
                self.save(f'./models/checkpoint{self.minibatch_steps+1}.pt')

    def save(self, PATH):
        torch.save({
            'cur_nnet': self.cur_nnet.state_dict(),
            'best_nnet': self.best_nnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr': self.learn_rate,
            'steps': self.minibatch_steps+1
        }, PATH)

    def load(self, PATH):
        checkpoint = torch.load(PATH,map_location=self.device)
        self.cur_nnet.load_state_dict(checkpoint['cur_nnet'])
        self.best_nnet.load_state_dict(checkpoint['best_nnet'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.learn_rate = checkpoint['lr']
        self.minibatch_steps = checkpoint['steps']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',type=str)
    parser.add_argument('num',type=int)
    
    args = parser.parse_args()
    # print(args)
    Coach = TrainPipeline()
    if args.model:
        Coach.load(args.model)
    Coach.run(args.num)