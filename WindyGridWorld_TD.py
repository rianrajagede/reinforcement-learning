import numpy as np
import itertools

from collections import defaultdict
from WindyGridWorld import WindyGridWorld
from lib import plotting

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
        
        env.reset()
        now_state = env.state()
        
        # Chosen action
        act_prob = policy(Q, epsilon, now_state)
        action = np.random.choice(np.arange(len(act_prob)), p=act_prob)

        # Generate one episode
        for t in itertools.count():
            
            # Take action
            next_state, done, reward = env.act(action)
            
            # Update statistics
            stats.episode_rewards[e] += reward
            stats.episode_lengths[e] = t

            # Get next action
            next_act_prob = policy(Q, epsilon, next_state)
            next_action = np.random.choice(np.arange(len(next_act_prob)),
                                                           p=next_act_prob)

            # Not waiting to generate one episode for TD update
            td_target = reward + discount * Q[next_state][next_action]
            td_delta = td_target - Q[now_state][action]
            Q[now_state][action] += alfa * td_delta

            if done:
                break
            
            # Move to the next state
            now_state = next_state
            action = next_action

    return Q, stats

"""Block code below optimize Q and policy then simulate it
"""
print "TEMPORAL-DIFFERENCE CONTROL A.K.A SARSA OPTIMIZE THE POLICY AND Q-VALUE"

Q, stats = td_control(policy_epsilon, n_episodes=200)
plotting.plot_episode_stats(stats)

print "SIMULATE THE OPTIMIZED POLICY AND Q-VALUE"

env.reset()
state = env.state()
done = False
step = 0
while not done:
    step += 1
    action = np.argmax(Q[state])
    next_state, done, reward = env.act(action)
    print "action: "+str(action)
    state = next_state
    print state
print "Total Step: "+str(step)