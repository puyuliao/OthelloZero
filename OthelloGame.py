from OthelloLogic import set_initial_board, get_possible_move, resolve_move, popcount, print_board
import numpy as np
import torch

def bitboard_to_list(bitboard):
    ls = list(map(int, '{:064b}'.format(bitboard)))
    ls.reverse()
    return ls

class Game():
    def __init__(self, device):
        self.device = device
        self.set_initial_state()
        self.Draw = 0.1
        self.boardsize = 8
    def set_initial_state(self):
        self.p1piece, self.p2piece = set_initial_board()
        self.cur_player = 1 # 1 for p1, -1 for p2

    # get_vaild_moves(self)
    # return a numpy array for all possible moves for cur_player
    def get_vaild_moves(self):
        vaild_lists = bitboard_to_list(get_possible_move(self.p1piece, self.p2piece))
        if sum(vaild_lists) == 0: 
            vaild_lists.append(1) # running out of moves and can only pass
        else:
            vaild_lists.append(0) # not running out of moves and cannot pass
        return np.array(vaild_lists)

    # get_nnet_format(self)
    # get neural network input : [channal_size=4, boardsize=8, boardsize=8]
    def get_nnet_format(self):
        mypiece = bitboard_to_list(self.p1piece)
        oppiece = bitboard_to_list(self.p2piece)
        myvaild = bitboard_to_list(get_possible_move(self.p1piece, self.p2piece))
        opvaild = bitboard_to_list(get_possible_move(self.p2piece, self.p1piece))
        return torch.from_numpy(np.array(mypiece + oppiece + myvaild + opvaild,dtype=np.float32).reshape((1,-1,8,8))).to(self.device)

    def get_numpy_format(self):
        mypiece = bitboard_to_list(self.p1piece)
        oppiece = bitboard_to_list(self.p2piece)
        myvaild = bitboard_to_list(get_possible_move(self.p1piece, self.p2piece))
        opvaild = bitboard_to_list(get_possible_move(self.p2piece, self.p1piece))
        return np.array(mypiece + oppiece + myvaild + opvaild,dtype=np.float32).reshape((-1,8,8))

    def set_next_state(self, move_id): # 64 stand for pass
        move_id = int(move_id)
        if 0 <= move_id < 64:
            self.p1piece, self.p2piece = resolve_move(self.p1piece, self.p2piece, move_id)
        # change board to a view form current player side 
        self.cur_player = -self.cur_player
        self.p1piece, self.p2piece = self.p2piece, self.p1piece
    
    def get_game_result(self):
        # return 0 if not ended, 1 if cur_player win, -1 if cur_player loss, 0.1 for draw
        if get_possible_move(self.p1piece,self.p2piece) > 0:
            return 0
        if get_possible_move(self.p2piece,self.p1piece) > 0:
            return 0
        diff = popcount(self.p1piece) - popcount(self.p2piece)
        if diff > 0: 
            return 1
        elif diff < 0:
            return -1
        else:
            return self.Draw 

    def print_game_state(self):
        print(self.cur_player)
        if self.cur_player == 1:
            print_board(self.p1piece, self.p2piece)
            print(f'{popcount(self.p1piece)} - {popcount(self.p2piece)}')
        else:
            print_board(self.p2piece, self.p1piece)
            print(f'{popcount(self.p2piece)} - {popcount(self.p1piece)}')
        print(f'result: {self.get_game_result()}')

if __name__ == "__main__":
    game = Game()
    
    game.print_game_state()
    while game.get_game_result() == 0:
        acts = np.nonzero(game.get_vaild_moves())[0]
        print(acts)
        action = int(np.random.choice(acts))
        print('currerent ',game.cur_player)
        print('action ',action)
        game.set_next_state(action)
        game.print_game_state()
        print()
