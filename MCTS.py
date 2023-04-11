"""
Mathilde Jeannin, 2023
jeannin@lix.polytechnique.fr 
Based on Luke Harold Miles's work. Check : https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""

import random 
import math
from abc import ABC, abstractmethod
from collections import defaultdict

class MCTS: 

    def __init__(self, two_players):
        self.children = dict() ## children[node] = node classes of all children of given node
        self.rewards = defaultdict(int) ## reward[node] = reward of this node
        self.visit_count = defaultdict(int) ## visit_count[node] = number of visit of this node
        self.exploration_weight = 1
        self.two_players = two_players

    def choose(self, node):
        # une fois que l'arbre a été visité suffisamment de fois, on utilise choose pour choisir l'action qui a le plus de potentielle 
        children = self.children[node]
        choosen_node = random.choice(children)
        max_reward = 0
        for child in children: 
            if self.rewards[child] > max_reward:
                max_reward = self.rewards[child]
                choosen_node = child
        return choosen_node

    def do_rollout(self, node):
        # rollour à partir de l'état actuel (node ici = état actuel)
        path = self._select_node(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulation(leaf)
        self._backpropagation(path, reward)

    def _select_node(self, node):
        # On cherche un noeud encore inexploré en enregistrant ses parents
        path=[]
        while True :
            path.append(node)
            if node not in self.children or not self.children[node]:
                return path # node unexplored or terminal
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)

    def _uct_select(self, node):
        assert all(n in self.children for n in self.children[node])
        # on verifie que tous les enfants de node font partis de self.children
        # i.e. ont déjà été expand

        # choix random par défault pour le cas ou encore aucun enfant n'a été visité. 
        # Si aucune visite, alors pour tous les enfants UCT = +infini, donc on randomise ce cas la 
        
        log_visit_parent = 2*math.log(self.visit_count[node])
        
        def _uct(node):
            explo = self.exploration_weight * math.sqrt((log_visit_parent)/self.visit_count[node])
            exploit = self.rewards[node]/self.visit_count[node]
            return exploit + explo
        
        selected_node = None
        max_uct = -10000000

        for child in self.children[node]:
            uct = _uct(child)
            if uct > max_uct:
                max_uct = uct
                selected_node = child
        # $$
        # selected_node = max(children, key=uct) 
        # -> si plusieurs max il prend le 1er de la liste donc pas random. i.e. bien garder la ligne selected_node = random.choice(children) pour definir la variable
        # print(max_uct, self.rewards[node], selected_node.board, selected_node.position)
        return selected_node

    def _simulation(self, node):
        # hypothèse que l'arbre n'est pas infini ni trop profond. A modifier pour des jeux/simu infini pour prendre en compte un horizon
        # on utilise les fonctions de la classe Node parce qu'on va dans un morceau d'arbre pas créer 
        if self.two_players:
            invert_reward = True
            while True:
                if node.is_terminal():
                    reward = node.reward()
                    return 1 - reward if invert_reward else reward
                node = random.choice(node.all_possible_children())
                invert_reward = not invert_reward
        if not self.two_players:
            while True:
                if node.is_terminal():
                    reward = node.reward()
                    return reward
                node = random.choice(node.all_possible_children())

    
    def _backpropagation(self, path, reward):
        ## back propagate the reward up to all parents until root
        # is_root = self.children[node][0]
        # while not is_root:
        #     self.rewards[node] += reward
        #     self.visit_count[node] += 1
        #     node = self.children[node][1]
        #     is_root = self.children[node][0]
        for node in reversed(path):
            self.rewards[node]+=reward
            self.visit_count[node]+=1
            if self.two_players:
                reward = 1 - reward # 1 pour moi = 0 pour l'adversaire, reste 0.5 en cas d'egalite

    def _expand(self, node):
        all_children = node.all_possible_children()
        if node in self.children:
            return # already expanded
        self.children[node]=all_children


class Node(ABC):

    @abstractmethod
    def all_possible_children(self):
        ## find all possible children
        return set()
    
    @abstractmethod
    def is_terminal(self):
        ## if it is end of the game
        return False
    
    @abstractmethod
    def reward(self):
        ## define win, lose, draw and reward if node is terminal
        return 0
    
    @abstractmethod
    def make_move(self):
        # return a Node class
        return Node