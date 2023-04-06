from MCTS import MCTS, Node
import math
import random 

'''
explo grille de 5x5 comme morpion (plus ?)
idee : quand l'agent passe par une case, met une X ? 
ajout d'obstacle O randomisé ? 
'''

class ExploBoard(Node):

    def __init__(self, size, board):
        self.size = size 
        self.board = board
        return 
    
    def all_possible_children(self):
        return super().all_possible_children()
    
    def is_terminal(self):
        return super().is_terminal()
    
    def reward(self):
        return super().reward()
    
    def make_move(self):
        return super().make_move()
    
    def to_pretty_string(self):
        to_char = lambda v: ("X" if v is True else ("O" if v is False else " "))
        rows = [
            [to_char(self.board[self.size * row + col]) for col in range(self.size)] for row in range(self.size)
        ]
        pretty_column = "\n " + str()
        return (
            "  " + " ".join(str(i+1) for i in range(self.size)) + "\n"
            + "\n".join(str(i + 1) + " " + " ".join(row) for i, row in enumerate(rows))
            + "\n"
        )
    
size=5
# board = (None,)*size**2
Board = ()
x,y = [],[]
for i in range(25):
    x.append(random.randint(0,25))
    y.append(random.randint(0,25))
for i in range(size**2):
    if i in x:
        Board+= (True,)
    elif i in y:
        Board+= (False,)
    else:
        Board+=(None,)

if __name__ == "__main__" :
    explo = ExploBoard(size,Board)
    print(explo.to_pretty_string())
