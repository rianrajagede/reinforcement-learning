UP = 0
LEFT = 1
DOWN = 2
RIGHT = 3

class WindyGridWorld(object):
    """Create an environment of a Grid World
    R = dict {(s):reward} - reward matrix, reward which obtained on state s
    """

    def __init__(self, shape=(7, 10),  start=(3, 0), terminate=(3, 7),
                    wind=[0, 0, 0, 1, 1, 1, 2, 2, 1, 0]):
        self.shape = shape
        self.R = {}
        self.terminate = terminate
        self.start = start
        self.now = start
        self.wind = wind

        for i in xrange(self.shape[0]):
            for j in xrange(self.shape[1]):
                self.R[(i, j)] = 0 if (i, j)==self.terminate else -1.0
                       
    def state(self):
        return self.now
    
    def reset(self):
        self.now = (3, 0)

    def act(self, action):
        if action==UP:
            self.act_up()
        if action==RIGHT:
            self.act_right()
        if action==LEFT:
            self.act_left()
        if action==DOWN:
            self.act_down()
        self.act_up(step=self.wind[self.now[1]])
        return (self.state(), self.state()==self.terminate, self.R[self.state()])

    def act_up(self, step=1):
        self.now = (max(self.now[0]-step, 0), self.now[1])

    def act_down(self, step=1):
        self.now = (min(self.now[0]+step, self.shape[0]-1), self.now[1])

    def act_right(self, step=1):
        self.now = (self.now[0], min(self.now[1]+step, self.shape[1]-1))

    def act_left(self, step=1):
        self.now = (self.now[0], max(self.now[1]-step, 0))
