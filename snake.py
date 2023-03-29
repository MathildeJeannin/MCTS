"""
Mathilde Jeannin, 2023
jeannin@lix.polytechnique.fr 
"""
import numpy as np
import random
from MCTS import MCTS, Node

class Snake(Node):
    def __init__(self, board, terminal, size, loose):
        # terminal : simulation s'arrete si perdu (loose), ou manger pomme
        # loose = perdu (s'est mangé)
        self.board = board #tuple t.q. board[0] -> position corps (board[0][0] = tete serpent), board[1] = position pomme
        self.terminal = terminal
        self.size = size
        self.loose = loose
        # la pomme a ete mange au coup precedent

    def all_possible_children(self):
        if self.is_terminal:
            []
        moves = []
        for direction in np.array([[0,1],[0,-1], [-1,0],[-1,0]]): 
            if not (self.board[0][1]-self.board[0][0]==direction).all() and not (self.board[0][0]+direction>np.array(self.size)).any() and not (self.board[0][0]+direction<np.array([0,0])).any():
                move = self.make_move(direction)
                moves.append(move)
        print(self.board)
        return moves
    
    def is_terminal(self):
        return self.terminal
    
    def reward(self):
        if not self.terminal:
            raise RuntimeError(f"reward called on nonterminal board {self.board}")
        if self.loose == None: ## n'a fait que des move autorisés sans ramasser de pomme donc pas terminal
            raise RuntimeError(f"reward called on nonterminal board {self.board}")
        if self.loose: ## s'est mangé
            return -10
        if not self.loose: ## a manger une pomme
            return 5
        return 
    
    def make_move(self, direction):
        loose = None
        terminal = False
        # tup = ([self.board[0][0]+direction],self.board[1])
        body = np.array([self.board[0][0]+direction])
        apple = self.board[1]
        coord_prec = self.board[0][0]
        for coord in self.board[0][1:]:
            body = np.concatenate((body, [coord_prec]))
            coord_prec=coord
        for body_part in self.board[0][1:]: #nouvel emplacement tete dans nouvel emplacement corps
            if body_part[0]==body[0][0] and body_part[1]==body[0][1]:
                terminal = True
                loose = True
        if (body[0]==self.board[1]).all(): #manger une pomme
            apple = newapple(self.size, body)
            terminal = True
            loose = False
        return Snake(board=(body, apple), terminal=terminal, size=self.size, loose=loose)
    
def newapple(size, snake):
    find_position = False
    while not find_position:
        x = random.randint(1,size[0])
        y = random.randint(1,size[1])
        if x not in snake or y not in snake:
            find_position = True
    return np.array([x,y])


## recreer un arbre a chaque pomme mangee ? 

def play_game():
    size = [50,50]
    head = np.array([random.randint(3,size[0]-3),random.randint(3,size[0]-3)])
    direction = np.array([[0,1],[0,-1],[1,0],[-1,0]])
    body = np.concatenate(([head], [head+random.choice(direction)]))
    two_players = False
    apple = newapple(size, body)
    tree = MCTS(two_players=two_players)
    snake = Snake((body,apple), False, size, False)
    while True:
        for _ in range(100):
            tree.do_rollout(snake)
        snake = tree.choose(snake)
        if snake.loose:
            break

if __name__=="__main__":
    play_game()