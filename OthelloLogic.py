# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 14:33:33 2020

@author: puyul
"""

# Reference from:
# https://www.hanshq.net/othello.html#bitboards
# http://www.cs.cmu.edu/~mmaxim/211-s03/bitboards.html

# int object are immutable in python 
# mypiece and oppiece are both size64 bit-array

# If no legal move, the current player should pass.
# The constant only use for state representation (for mcts).
CONST_PASS = 128

def set_disk(mypiece,x,y):
    return mypiece | (1 << (x*8 + y))

# set_initial_board(uint64, uint64)
# Set the initial bit-board position. 
# bit-board format:
# 00 01 02 03 04 05 06 07
# 08 09 10 11 12 13 14 15 
# 16 17 18 19 20 21 22 23
# 24 25 26 27 28 29 30 31
# 32 33 34 35 36 37 38 39
# 40 41 42 43 44 45 46 47
# 48 49 50 51 52 53 54 55
# 56 57 58 59 60 61 62 63
def set_initial_board():
    mypiece = oppiece = 0
    mypiece = set_disk(mypiece,3,3)
    mypiece = set_disk(mypiece,4,4)
    oppiece = set_disk(oppiece,3,4)
    oppiece = set_disk(oppiece,4,3)
    return mypiece, oppiece

def print_board_on_colab(mypiece,oppiece):    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for i in range(1,9):
        for j in range(1,9):
            ax.add_patch(plt.Rectangle((i, j), 1, 1, edgecolor='black',
            facecolor='green', linewidth=2))

    for i in range(8):
        plt.text(i+1.35, 0, chr(65+i))

    for i in range(8):
        plt.text(0, i+1.35, chr(56-i))

    for i in range(8):
        for j in range(8):
            idx = 1 << i*8+j
            if mypiece & idx:
                ax.add_patch(plt.Circle((j+1.5, i+1.5), 0.39, edgecolor='black',
                facecolor='black', linewidth=2))
            elif oppiece & idx:
                ax.add_patch(plt.Circle((j+1.5, i+1.5), 0.39, edgecolor='white',
                facecolor='white', linewidth=2))

    ax.set(xticks=[], yticks=[])
    ax.axis('image')
    ax.grid()
    plt.show()


def print_board(mypiece,oppiece):
    print('\ABCDEFGH')
    for i in range(8):
        print(i+1,end='')
        for j in range(8):
            idx = 1 << (7-i)*8+j
            if mypiece & idx:
                print('@',end='')
            elif oppiece & idx:
                print('O',end='')
            else:
                print('.',end='')
        print()

# popcount(uint64)
# Count the number of 1 in the binary of the number.
PRE_CALC_POPCNT = []
def popcount(mypiece):
    global PRE_CALC_POPCNT
    if len(PRE_CALC_POPCNT) == 0:
        for i in range(0x10000):
            PRE_CALC_POPCNT.append(bin(i).count('1'))
    return PRE_CALC_POPCNT[mypiece&0xffff] + PRE_CALC_POPCNT[mypiece>>16&0xffff] + \
            PRE_CALC_POPCNT[mypiece>>32&0xffff] + PRE_CALC_POPCNT[mypiece>>48&0xffff]

NUM_DIRS = 8
#R DR D DL L UL U UR
MASKS = [0x7F7F7F7F7F7F7F7F,0x007F7F7F7F7F7F7F,0xFFFFFFFFFFFFFFFF,0x00FEFEFEFEFEFEFE,
         0xFEFEFEFEFEFEFEFE,0xFEFEFEFEFEFEFE00,0xFFFFFFFFFFFFFFFF,0x7F7F7F7F7F7F7F00]
LSHIFTS = [0,0,0,0,1,9,8,7]
RSHIFTS = [1,9,8,7,0,0,0,0]

def shift(mypiece, sdir):
    global NUM_DIRS
    global MASKS
    global LSHIFTS
    global RSHIFTS
    if sdir < (NUM_DIRS >> 1):
        return (mypiece >> RSHIFTS[sdir]) & MASKS[sdir]
    return (mypiece << LSHIFTS[sdir]) & MASKS[sdir]

# get_possible_move(uint64, uint64)
# Output a uint64 represents all possible move of mypiece in bit-board representation.
def get_possible_move(mypiece, oppiece):
    global NUM_DIRS
    empty_disks = ~(mypiece | oppiece)
    legal_moves = 0
    for sdir in range(NUM_DIRS):
        x = shift(mypiece, sdir) & oppiece
        x |= shift(x, sdir) & oppiece
        x |= shift(x, sdir) & oppiece
        x |= shift(x, sdir) & oppiece
        x |= shift(x, sdir) & oppiece
        x |= shift(x, sdir) & oppiece
        legal_moves |= shift(x,sdir) & empty_disks
    return legal_moves

def is_move(mypiece, oppiece, r, c):
    mask = 1 << (r*8+c)
    return (get_possible_move(mypiece, oppiece) & mask) != 0

# resolve_move(uint64, uint64, move_postion), 0 <= move_postion < 64 
# Perform a move and return the next state in bit-board representation.
def resolve_move(mypiece, oppiece, boardidx):
    global NUM_DIRS
    captured_disks = 0
    new_disk = 1 << boardidx
    mypiece |= new_disk
    for sdir in range(NUM_DIRS):
        x = shift(new_disk, sdir) & oppiece
        x |= shift(x, sdir) & oppiece
        x |= shift(x, sdir) & oppiece
        x |= shift(x, sdir) & oppiece
        x |= shift(x, sdir) & oppiece
        x |= shift(x, sdir) & oppiece
        bounding_disk = shift(x, sdir) & mypiece
        if bounding_disk != 0: 
            captured_disks |= x
    mypiece ^= captured_disks
    oppiece ^= captured_disks
    return mypiece, oppiece

if __name__ == '__main__':
    import random
    a = b = cnt = 0
    a,b = set_initial_board()
    print_board(a,b)
    while get_possible_move(a,b) or get_possible_move(b,a):
        if cnt & 1: x = get_possible_move(a,b)
        else: x = get_possible_move(b,a)
        print("x ",x)
        if x:
            temp = []
            for i in range(64):
                if x>>i&1: temp.append(i)
            print("temp ",temp)
            if cnt & 1: a,b = resolve_move(a,b,temp[random.randint(0,len(temp)-1)])
            else: b,a = resolve_move(b,a,temp[random.randint(0,len(temp)-1)])
        else: print("skiped")
        print_board(a,b)    
        print()
        cnt ^= 1
    print(popcount(a),popcount(b))