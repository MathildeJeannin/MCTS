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

    def __init__(self, size, board, position, terminal, reward):
        self.row = size[0] ## ex : grille de 4x6 size=[4,6]
        self.col = size[1]
        self.board = board ## tuple de taille size[0]*size[1]
        self.terminal = terminal 
        self.position = position ## position du robot de la forme (row-1)*size[0]+col ? ou [row,col] ?
        self.actual_reward = reward
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
                    # print(future_position)
                ## pas possible si :
                ##      on veut aller à droite mais on est déjà tout à droite
                ##      on veut aller à gauche mais on est déjà tout à gauche 
                    future_move = self.make_move(future_position)
                    # print("future move:")
                    # print(future_move.to_pretty_string())
                    moves.append(future_move)
        return moves
    
    def is_terminal(self):
        return self.terminal
    
    def reward(self):
        return self.actual_reward
    
    def make_move(self, index):
        visit = True 
        reward = self.actual_reward
        if self.board[index]==False: ## obstacle
            reward += -50
            visit = False ##reste un obstacle
        elif self.board[index]==None: ## pas visité
            reward += 20
        elif self.board[index]==True:
            reward += 1
        tup = self.board[:index] + (visit,) + self.board[index+1:]
        is_terminal = not None in tup
        # print(is_terminal)
        if is_terminal:
            reward+=50
        return ExploBoard(size = [self.row, self.col], board = tup, position = index, terminal = is_terminal, reward = reward)
    
    # def to_pretty_string(self):
    #     to_char = lambda v: ("O" if v is True else ("X" if v is False else " "))
    #     rows = [
    #         [to_char(self.board[self.row * row + col]) for col in range(self.col)] for row in range(self.row)
    #     ]
    #     pretty_column = "\n " + str()
    #     return (
    #         "  " + " ".join(str(i+1) for i in range(self.col)) + "\n"
    #         + "\n".join(str(i + 1) + " " + " ".join(row) for i, row in enumerate(rows))
    #         + "\n"
    #     )

    def to_pretty_string(self):
        pretty_grid = np.empty((self.row,self.col),str)
        i = 0
        for i,grid in enumerate(self.board):
            r,c = int(i/self.row),i%self.row
            if grid == True and i != self.position: ##deja visite, pas la pose actuelle
                pretty_grid[r,c] = "."
            elif grid == False and i != self.position: ## obstacle et pas la pose actuelle
                pretty_grid[r,c] = "X"
            elif i == self.position: ## pose actuelle
                pretty_grid[r,c] = "O"
            else: ## pas visité, pas un obstacle, pas la pose actuelle 
                pretty_grid[r,c] = " "
            i+=1
        return (
            "  " + " ".join(str(i+1) for i in range(self.col)) + "\n"
            + "\n".join(str(i + 1) + " " + " ".join(row) for i, row in enumerate(pretty_grid))
            + "\n"
        )
    
# size=[4,7]
# # board = (None,)*size[0]*size[1]
# Board = ()
# x,y = [],[]
# for i in range(size[0]*size[1]):
#     x.append(random.randint(0,size[0]*size[1]))
#     y.append(random.randint(0,size[0]*size[1]))
# for i in range(size[0]*size[1]):
#     if i in x:
#         Board+= (True,)
#     elif i in y:
#         Board+= (False,)
#     else:
#         Board+=(None,)

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
    explo = ExploBoard(size=size, board=board, position=pose_init, terminal=False,reward=0)
    print(explo.to_pretty_string())
    i = 0
    while True:
        for _ in range(100):
            tree.do_rollout(explo)
        explo = tree.choose(explo)
        print(i)
        print(explo.to_pretty_string())
        print(explo.board, explo.position)
        if explo.terminal:
            break
        i+=1


if __name__ == "__main__" :
    explore_board([5,5],3)
