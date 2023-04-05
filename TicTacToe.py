"""
Mathilde Jeannin, 2023
jeannin@lix.polytechnique.fr 
Based on Luke Harold Miles's work. Check : https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""

import random 
import math
from MCTS import MCTS, Node
import numpy as np

class TicTacToe(Node):

    def __init__(self, board, turn, winner, terminal):
        self.board = board
        # None = pas joué; True = jouer par le joueur, False = jouer par l'ordi
        self.turn = turn #True si au joueur de jouer
        self.winner = winner #True si gagner par le jouer, False si gagner par l'ordi
        self.terminal = terminal 

    def all_possible_children(self):
        if self.terminal:  # If the game is finished then no moves can be made
            return []
        # Otherwise, you can make a move in each of the empty spots
        moves = []
        for i in range(9): 
            if self.board[i]==None:
                move = self.make_move(i)
                moves.append(move)
        return moves

    
    def is_terminal(self):
        return self.terminal
    
    def reward(self):
        if not self.terminal:
            raise RuntimeError(f"reward called on nonterminal board {self.board}")
        if self.winner is self.turn:
            # It's your turn and you've already won. Should be impossible.
            raise RuntimeError(f"reward called on unreachable board {self.board}")
        if self.turn is (not self.winner):
            return 0  # Your opponent has just won. Bad.
        if self.winner is None:
            return 0.5  # Board is a tie
        # The winner is neither True, False, nor None
        raise RuntimeError(f"board has unknown winner type {self.winner}")
         
    def make_move(self, index):
        if self.board[index]!=None:
            raise RuntimeError("Invalid move")
        tup = self.board[:index] + (self.turn,) + self.board[index+1:]
        turn = not self.turn
        winner = _find_winner(tup)
        is_terminal = (winner is not None) or not any(v is None for v in tup)
        return TicTacToe(board=tup, turn=turn, winner=winner, terminal=is_terminal)
    

    def to_pretty_string(self):
        to_char = lambda v: ("X" if v is True else ("O" if v is False else " "))
        rows = [
            [to_char(self.board[3 * row + col]) for col in range(3)] for row in range(3)
        ]
        return (
            "\n  1 2 3\n"
            + "\n".join(str(i + 1) + " " + " ".join(row) for i, row in enumerate(rows))
            + "\n"
        )

def _find_winner(tup):
    "Returns None if no winner, True if X wins, False if O wins"
    for i1, i2, i3 in _winning_combos():
        v1, v2, v3 = tup[i1], tup[i2], tup[i3]
        if False is v1 is v2 is v3:
            return False
        if True is v1 is v2 is v3:
            return True
    return None
    
def _winning_combos():
    for start in range(0, 9, 3):  # three in a row
        yield (start, start + 1, start + 2)
    for start in range(3):  # three in a column
        yield (start, start + 3, start + 6)
    yield (0, 4, 8)  # down-right diagonal
    yield (2, 4, 6)  # down-left diagonal


def play_game():
    turn = True
    two_players = True
    tree = MCTS(two_players=two_players)
    TTC = TicTacToe((None,)*9, turn, False, False)
    print(TTC.to_pretty_string())
    while True:
        if turn :
            row_col = input("enter row,col: ")
            row, col = map(int, row_col.split(","))
            index = 3 * (row - 1) + (col - 1)
            if TTC.board[index] is not None:
                raise RuntimeError("Invalid move")
            TTC = TTC.make_move(index)
            print(TTC.to_pretty_string())
            turn = False
            if TTC.terminal:
                break
        # You can train as you go, or only at the beginning.
        # Here, we train as we go, doing fifty rollouts each turn.
        for _ in range(1000):
            tree.do_rollout(TTC) #-> un noeud = un etat
        TTC = tree.choose(TTC)
        print(TTC.to_pretty_string())
        turn = True
        if TTC.terminal:
            break

def play_game_alone():
    turn = True
    two_players = True
    tree = MCTS(two_players=two_players)
    TTC = TicTacToe((None,)*9, turn, False, False)
    print(TTC.to_pretty_string())
    while True:
        if turn :
            for _ in range(1000):
                tree.do_rollout(TTC)
            TTC = tree.choose(TTC)
            print(TTC.to_pretty_string())
            turn = False
            if TTC.terminal:
                break
        for _ in range(1000):
            tree.do_rollout(TTC) #-> un noeud = un etat
        TTC = tree.choose(TTC)
        print(TTC.to_pretty_string())
        turn = True
        if TTC.terminal:
            break

def new_tic_tac_toe_board():
    return TicTacToe()


if __name__ == "__main__":
    play_game()
