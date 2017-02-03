import numpy as np

from collections import defaultdict
from WindyGridWorld import WindyGridWorld
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


env = WindyGridWorld()

def policy_epsilon(Q, epsilon, state):
    """Policy-epsilon :
	   - epsilon probability choosing random action
		- 1-epsilon probability choosing action wich has maximum Q value
    """
    if np.random.rand() <= epsilon:
        # Explore
        return np.array([.25, .25, .25, .25])
    else:
        # Exploit
        best_action = np.argmax(Q[state])
        A = np.zeros(4)
        A[best_action] = 1
        return A

# td_control a.k.a SARSA
def td_control(policy, n_episodes, alfa=0.5, epsilon=0.1, discount=1.0, env=env):

    # Make a dictionary with deafult value 0.0
    Q = defaultdict(lambda: np.zeros(4))
    
    # for Denny Britz's plotting
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(n_episodes),
        episode_rewards=np.zeros(n_episodes))

    for e in xrange(n_episodes):
        print "\rEpisode " + str(e + 1) 
        now_state = env.state()
        terminate = False

        # Chosen action
        act_prob = policy(Q, epsilon, now_state)
        action = np.random.choice(np.arange(len(act_prob)), p=act_prob)

        # Generate one episode
        step = 0
        while not terminate:
            
            # Take action
            next_state, done, reward = env.act(action)
            
            # Update statistics
            stats.episode_rewards[e] += reward
            stats.episode_lengths[e] = step

            # Get next action
            next_act_prob = policy(Q, epsilon, now_state)
            next_action = np.random.choice(np.arange(len(next_act_prob)),
                                                           p=next_act_prob)

            # Not waiting to generate one episode for TD update
            delta = reward + discount * Q[next_state][next_action] \
                                                - Q[now_state][action]
            Q[now_state][action] = Q[now_state][action] + alfa * delta

            # Move to the next state
            now_state = next_state
            action = next_action
            step = step + 1
            if done:
                terminate = True

    return Q, stats

"""Block code below optimize Q and policy then run it
"""
print "TEMPORAL-DIFFERENCE CONTROL A.K.A SARSA OPTIMIZE THE POLICY AND Q-VALUE"

Q, stats = td_control(policy_epsilon, n_episodes=200)
plotting.plot_episode_stats(stats)

#print "SIMULATE THE OPTIMIZED POLICY AND Q-VALUE"
#
#env.reset()
#state = env.state()
#done = False
#while not done:
#    action = np.argmax(Q[state])
#    next_state, done, reward = env.act(action)
#    print "action: "+str(action)
#    state = next_state
#    print state
