from MCTS import MCTS, Node
import math
import random 
import numpy as np

'''
explo grille de 5x5 comme morpion (plus ?)
idee : quand l'agent passe par une case, met un O ? ou enlève un O
ajout d'obstacle X randomisé ? 
Obstacle : False X
Visité : True O
Non visité et pas d'obstacle : None
Position robot : coord dans la grille --> gerer cas pose robot = pose obstacle
'''

class ExploBoard(Node):

    def __init__(self, size, board, position, terminal):
        self.row = size[0] ## ex : grille de 4x6 size=[4,6]
        self.col = size[1]
        self.board = board ## tuple de taille size[0]*size[1]
        self.terminal = terminal 
        self.position = position ## position du robot de la forme (row-1)*size[0]+col 
        return 
    
    def all_possible_children(self):
        if self.terminal:
            return []
        moves = []
        list_moves = [1, -1, self.col, -self.col] ## gauche droite bas haut
        # print(self.to_pretty_string())
        for move in list_moves:
            future_position = self.position + move
            if future_position >= 0 and future_position < self.row*self.col: 
                if not (move==1 and ((self.position+1)%self.col)==0) and not (move==-1 and self.position%self.col==0):
                ## pas possible si :
                ##      on veut aller à droite mais on est déjà tout à droite
                ##      on veut aller à gauche mais on est déjà tout à gauche 
                    future_move = self.make_move(future_position)
                    moves.append(future_move)
        return moves
    
    def is_terminal(self):
        return self.terminal
    
    def reward(self):
        # print(self.board, self.position)
        if not self.terminal:
            raise RuntimeError(f"reward called on nonterminal board {self.board}")
        elif self.board[self.position]==False:
            # print("boom")
            return -5
        else:
            return 50
    
    def make_move(self, index):
        if self.board[index]==False: #obstacle
            is_terminal = True
            tup = self.board
        else : 
            tup = self.board[:index] + (True,) + self.board[index+1:]
            is_terminal = not None in tup
        return ExploBoard(size = [self.row, self.col], board = tup, position = index, terminal = is_terminal)

    def to_pretty_string(self):
        pretty_grid = np.empty((self.row,self.col),str)
        i = 0
        for i,grid in enumerate(self.board):
            r,c = int(i/self.row),i%self.row
            if grid == True and i != self.position: ##deja visite
                pretty_grid[r,c] = "."
            elif grid == False and i != self.position: ## obstacle et pas la pose actuelle
                pretty_grid[r,c] = "X"
            elif i == self.position: ## pose actuelle
                pretty_grid[r,c] = "R"
            else: ## pas visité, pas un obstacle, pas la pose actuelle 
                pretty_grid[r,c] = " "
            i+=1
        return (
            "  " + " ".join(str(i+1) for i in range(self.col)) + "\n"
            + "\n".join(str(i + 1) + " " + " ".join(row) for i, row in enumerate(pretty_grid))
            + "\n"
        )

def explore_board(size, max_obstacles):
    nb_obstacles = random.randint(1,max_obstacles)
    obstacles=[]
    grid_size = size[0]*size[1]
    for i in range(nb_obstacles):
        obstacles.append(random.randint(0, grid_size-1))
    pose_init = random.randint(0, grid_size-1)
    board = ()
    for j in range(size[0]*size[1]):
        if j in obstacles and j!= pose_init:  
            board += (False,)
        elif j == pose_init:
            board += (True,)
        else:
            board += (None,)
    tree = MCTS(two_players=False)
    explo = ExploBoard(size=size, board=board, position=pose_init, terminal=False)
    print(explo.to_pretty_string())
    i = 0
    while True:
        for _ in range(100):
            tree.do_rollout(explo)
        explo = tree.choose(explo)
        print(explo.to_pretty_string())
        if explo.terminal:
            break
        i+=1


if __name__ == "__main__" :
    explore_board([5,5],1)