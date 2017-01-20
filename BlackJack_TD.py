# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 21:56:33 2017

@author: RianAdam
"""

import numpy as np

from collections import defaultdict
from lib import plotting

deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

class BlackJack(object):
    """Create an environment of a Black Jack Game
    
    dealer = list - dealer list card in his deck
    player = list - player list card in his deck
    done = bool - True if the game is done, and False otherwise
    
    this class only can be accessed from act(), reset(), state()
    
    act() = PARAMS : 1 if hit, 0 if stick
            RETURN : (state), done_status, reward 
    
    reset() = PARAMS : None
              RETURN : None
    
    state() = PARAMS : None
             RETURN : (state)
             
    (state) is a tuple of player score, dealers score, usable ace condition
    
    Black Jack Refferences:
    [1] https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node51.html (Example 5.1)
    [2] http://www.bicyclecards.com/how-to-play/blackjack/
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.dealer = [self.draw()]
        self.player = [self.draw()]
        self.done = False
                     
    def natural(self,hand): # check if he got natural/blackjack condition
        return sorted(hand)==[1,10]
                       
    def draw(self): # get one card
        return np.random.choice(deck)
    
    def usable(self, hand): # check if he got usable ace condition
        return 1 in hand and sum(hand) + 10 <= 21
        
    def busted(self, hand): # check if he got busted
        return self.sum_hand(hand) > 21
    
    def sum_hand(self, hand):
        if self.usable(hand):
            return sum(hand) + 10
        else:
            return sum(hand)
        
    def state(self):
        return self.sum_hand(self.player), self.sum_hand(self.dealer), \
                    self.usable(self.player)
    
    def act(self, hit):
        if not self.done:
            if hit:
                self.hit()
                if self.busted(self.player):
                    self.done = True
                    return self.state(), self.done, -1
                else:
                    return self.state(), self.done, 0
            else:
                return self.stick()
                    
    def hit(self):
        self.player.append(self.draw())
                        
    def stick(self):
        self.done = True
        
        # Dealer doing hit while his score below 17
        # see refference [2]
        while self.sum_hand(self.dealer) < 17:
            self.dealer.append(self.draw())
        
        # player'll never get busted in here so just sum it
        player_score = self.sum_hand(self.player) 
        
        dealer_score = -1 if self.busted(self.dealer) else self.sum_hand(self.dealer)
        
        if self.natural(self.player) and self.natural(self.dealer):
            reward = 1
        elif self.natural(self.player):
            reward = 1.5
        elif dealer_score > player_score:
            reward = -1
        elif dealer_score < player_score:
            reward = 1
        else:
            reward = 0
    
        return self.state(), self.done, reward
        

env = BlackJack()    

def policy_0(pl_score, de_score, use_ace):
    """Policy-0 : always "hit" except we got score >= 20
    """
    
    # Using probability instead of actual act number for consistency
    return np.array([1.0, 0.0]) if pl_score >= 20 else np.array([0.0, 1.0])

def policy_epsilon(Q, epsilon, state):
    """Policy-epsilon :
	   - always hit when player score < 12
	   - otherwise:
		   - epsilon probability choosing random action
		   - 1-epsilon probability choosing action wich has maximum Q value
    """
    
    # Greedy choose hit when score < 12
    if state[0] < 12:
        return [0.0, 1.0]
        
    if np.random.rand() <= epsilon:
        # Explore
        return np.array([0.5, 0.5])
    else:
        # Exploit
        best_action = np.argmax(Q[state])
        A = np.zeros(2)
        A[best_action] = 1
        return A
    
def td_policy_evaluation(policy, n_episodes, alfa=1.0, discount=1.0, env=env):
    
    # Make a dictionary with deafult value 0.0
    V = defaultdict(float)
    
    for e in xrange(n_episodes):
        env.reset()
        now_state = env.state()
        terminate = False
        
        # Running an episode
        while not terminate:
            # Chosen action
            act_prob = policy(*now_state)
            action = np.random.choice(np.arange(len(act_prob)), p=act_prob)
            
            # Action
            next_state, done, reward = env.act(action)
            if done:
                terminate = True
    
            # Not waiting to generate one episode for TD update
            V[now_state] = V[now_state] + alfa * (reward + discount * V[next_state]
                                - V[now_state])
            
            # Move to the next state
            now_state = next_state       
            
    return V


"""Block code below evaluate a policy and return a plotted V-value
"""
print "TEMPORAL-DIFFERENCE EVALUATE POLICY_0"         

V = td_policy_evaluation(policy_0, n_episodes=100000)

# Delete state with player score below 12 to make it same with example
# Because we call V[next_state] in the end, to make the same plot
# we should delete some keys
new_V = defaultdict(float)
for key, data in V.iteritems():
    if key[0] >= 12 and key[1]<=11:
        new_V[key] = data

# Using plotting library from Denny Britz repo
plotting.plot_value_function(new_V, title="Policy_0 Evaluation")
            
