import numpy as np

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


"""Below is an example how to use black jack environment

Run this code in intrepreter and there are 2 actions:
    act(1)/act(0) : hit or stick
    reset() : start/reset the game
"""

env = BlackJack()

def print_state( pl_score, de_score, use_ace, reward=0):
    if env.done:
        print "== Game Over =="
        print "Reward: {}".format(reward)
    print "Player: {} | Dealer: {} | Usable Ace: {}".format(
                pl_score, de_score, use_ace)

    # You shouldn't print deck list
    print "Player Deck: {}".format(env.player)
    print "Dealer Deck: {}".format(env.dealer)

def act(hit, env=env):
    state, done, reward = env.act(hit)
    pl_score, de_score, use_ace = state
    print_state(pl_score, de_score, use_ace, reward)

def reset(env=env):
    env.reset()
    pl_score, de_score, use_ace = env.state()
    print_state(pl_score, de_score, use_ace)
