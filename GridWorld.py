import numpy as np

nAction = 4
UP = 0
LEFT = 1
DOWN = 2
RIGHT = 3

class GridWorld(object):
    """Create an environment of a Grid World

    shape = tuple (row, col) - size of the Grid World
    terminate = tuple (row, col) - goal state
    P = dict {(s):{a:{(s',a):prob}}} - transition matrix, the probability
              of arrival on state s' from state s using action a.
              the last key using (s',a) for handling same arrival to same state
              from different action

    R = dict {(s):reward} - reward matrix, reward which obtained on state s
    """

    def __init__(self, shape=(4, 4), terminate=[(3, 3)]):
        self.shape = shape
        self.P = {}
        self.R = {}
        self.terminate = terminate

        for i in xrange(self.shape[0]):
            for j in xrange(self.shape[1]):
                self.P[(i, j)] = {}
                for a in xrange(nAction):
                    if (i, j) in self.terminate:
                        self.P[(i, j)][a] = {(i, j, UP): 1, (i, j, LEFT): 1,
                                           (i, j, DOWN): 1, (i, j, RIGHT): 1}
                    else:
                        self.P[(i,j)][a] = {
                                (i, min(j+1, self.shape[1]-1), RIGHT): 1 if a==RIGHT else 0,
                                (i, max(j-1, 0), LEFT):1 if a==LEFT else 0,
                                (min(i+1, self.shape[0]-1), j, DOWN):1 if a==DOWN else 0,
                                (max(i-1, 0), j, UP):1 if a==UP else 0
                          }

                self.R[(i, j)] = 0 if (i, j) in self.terminate else -1


def value_to_array(V, shape=(4,4)):
    """dict V to array V"""
    array_V = np.zeros(shape)
    for key, value in V.iteritems():
        array_V[key[0]][key[1]]=value

    return array_V

def policy_to_array(policy, shape=(4,4)):
    """dict policy to array policy"""
    array_policy = np.zeros(shape)
    for key, action in policy.iteritems():
        best_a = 0
        for a  in xrange(nAction):
            if policy[key][a] > policy[key][best_a]:
                best_a = a

        array_policy[key[0]][key[1]] = best_a

    return array_policy

def value_to_policy(V, shape=(4,4)):
    """given V generate the policy using greedy method"""
    policy = {}
    for i in xrange(shape[0]):
        for j in xrange(shape[1]):
            policy[(i,j)] = {}
            best_v = -10000000
            best_a = 0

            if V[(max(i-1,0),j)] > best_v:
                best_v = V[(max(i-1, 0), j)]
                best_a = UP
            if V[(min(i+1, shape[0]-1), j)] > best_v:
                best_v = V[(min(i+1, shape[0]-1), j)]
                best_a = DOWN
            if V[(i, max(j-1, 0))] > best_v:
                best_v = V[(i, max(j-1, 0))]
                best_a = LEFT
            if V[(i, min(j+1, shape[1]-1))] > best_v:
                best_v = V[(i, min(j+1, shape[1]-1))]
                best_a = RIGHT

            for a in xrange(nAction):
                policy[(i,j)][a]=1 if a==best_a else 0

    return policy

def policy_0(shape=(4, 4)):
    """policy method return a dictionary

    policy = dictionary {s:{a:prob}} - a probability from state s
        choosing action a

    policy_0 has an equal probability
    """
    policy = {}
    for i in xrange(shape[0]):
        for j in xrange(shape[1]):
            policy[(i, j)] = {0: .25, 1: .25, 2: .25, 3: .25}

    return policy

def policy_1(shape=(4, 4)):
    """policy method return a dictionary

    policy = dictionary {s:{a:prob}} - a probability from state s
        choosing action a

    policy_1 has a random probability
    """
    policy = {}
    for i in xrange(shape[0]):
        for j in xrange(shape[1]):
            policy[(i, j)] = {}

            action_prob = np.random.rand(nAction)
            action_prob = action_prob / np.sum(action_prob)

            for a in xrange(nAction):
                policy[(i, j)][a] = action_prob[a]

    return policy

def policy_2(shape=(4, 4)):
    """policy method return a dictionary

    policy = dictionary {s:{a:prob}} - a probability from state s
        choosing action a

    policy_2 has an equal probability but only going to north and west
    """
    policy = {}
    for i in xrange(shape[0]):
        for j in xrange(shape[1]):
            policy[(i,j)] = {0: .5, 1: .5, 2: 0, 3: 0}

    return policy

def policy_evaluation(policy, env, discount=1, shape=(4,4), epsilon=0.00001):
    """policy evaluation from Sutton's Book

    V = tuple (row,vcol) - value of state s (row, col)
    newVal = scalar - for temporary new value of state s
    """
    # create 0 value function
    V = {(i,j): 0 for j in xrange(shape[1]) for i in xrange(shape[0])}

    while True:
        last_V = V.copy()
        eror = 0

        # Bellman Expected Equation
        for i in xrange(shape[0]):
            for j in xrange(shape[1]):
                new_val = 0
                for a in xrange(nAction):
                    for s_new in env.P[(i, j)][a]:
                        next_state = (s_new[0], s_new[1])
                        new_val += (policy[(i, j)][a] *
                                env.P[(i, j)][a][s_new] * (env.R[(i, j)] +
                                discount * V[next_state]))
                V[(i, j)] = new_val
                eror = max(eror, np.abs(V[(i, j)] - last_V[(i, j)]))

        if eror < epsilon:
            break

    return V

def policy_iteration(policy, env, shape=(4,4), discount=1):
    """policy iteration from Sutton's Book"""
    while True:

        # Evaluate/generate value V based on policy
        V = policy_evaluation(policy, env)
        last_policy = policy.copy()

        # Generate new policy based on generated value V
        policy = value_to_policy(V)

        # If converged
        if last_policy==policy:
            break

    return policy, V

def value_iteration(env, shape=(4,4), discount=1):
    """value iteration from Sutton's Book"""

    # Create random value function on every state
    V = {(i,j): 0 for j in xrange(shape[1]) for i in xrange(shape[0])}

    while True:
        last_V = V.copy()

        # Create new value function
        for i in xrange(shape[0]):
            for j in xrange(shape[1]):
                best_val = -1000000

                # choose only the best action
                for a in xrange(nAction):
                    new_val=0
                    for s_new in env.P[(i, j)][a]:
                        next_state = (s_new[0], s_new[1])
                        new_val += env.P[(i, j)][a][s_new] * (env.R[(i, j)] +
                                discount * V[next_state])
                    best_val = max(new_val, best_val)

                V[(i,j)] = best_val

        # If converged
        if last_V == V:
            break

    return V

if __name__ == "__main__":

    # Create policy
    policy = policy_0()

    print "Initial Policy"
    print policy_to_array(policy)

    # create environment
    env = GridWorld(terminate=[(0, 0), (3, 3)])

    final_policy, final_value = policy_iteration(policy, env)

    # Uncomment these two lines below, and comment the line above
    # to execute using Value Iteration

#    final_value = value_iteration(env)
#    final_policy = value_to_policy(final_value)

    print "Value Result"
    print value_to_array(final_value)
    print "Policy Result"
    print policy_to_array(final_policy)
