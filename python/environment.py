import numpy as np


class Environment(object):
    """
    An environment skeleton. Defaults to simple MAB
    H = 1, K=2, rewards are bernoulli, sampled from dirichlet([1,1]) prior.
    """

    def __init__(self):
        self.horizon = 1
        self.actions = [0,1]
        self.reward_dists = np.random.dirichlet(np.array([1,1]))
        self.reward = lambda x,a: np.random.binomial(1,self.reward_dists[a])

        self.transition = lambda x,a: "s"
        self.start = lambda: "s"

        self.state = None
        self.h = 0

    def start_episode(self):
        self.h=0
        self.state = self.start()
        return self.state

    def get_actions(self):
        if self.state == None:
            raise Exception("Episode not started")
        if self.h == self.horizon:
            return None
        return self.actions
    
    def act(self, a):
        if self.state == None:
            raise Exception("Episode not started")
        r = np.random.binomial(1, self.reward(self.state,a))
        self.h += 1
        if self.h == self.horizon:
            self.state = None
        else:
            self.state = self.transition(self.state,a)
        return(self.state, r)

class CombinationLock(Environment):
    def __init__(self, horizon=5):
        self.horizon=horizon
        self.actions = [0,1]
        self.opt = np.random.choice(self.actions,size=self.horizon)

    def transition(self, x, a):
        if x == None:
            raise Exception("Not in any state")
        if x[0] == "g" and a==self.opt[x[1]]:
            return("g",x[1]+1)
        return ("b",x[1]+1)
            
    def start(self):
        return ("g",0)

    def reward(self, x, a):
        if x == ("g",self.horizon-1) and a == self.opt[x[1]]:
            return 1
        return 0

class StochasticCombinationLock(Environment):
    def __init__(self,horizon=5, swap=0.1):
        self.horizon=horizon
        self.swap = swap
        self.actions = [0,1]
        self.opt_a = np.random.choice(self.actions,size=self.horizon)
        self.opt_b = np.random.choice(self.actions,size=self.horizon)

    def transition(self,x,a):
        if x == None:
            raise Exception("Not in any state")
        b = np.random.binomial(1,self.swap)
        if x[0] == "a" and a == self.opt_a[x[1]]:
            if b == 0:
                return(("a", x[1]+1))
            else:
                return(("b", x[1]+1))
        if x[0] == "b" and a == self.opt_b[x[1]]:
            if b == 0:
                return(("b", x[1]+1))
            else:
                return(("a", x[1]+1))
        else:
            return(("c", x[1]+1))

    def start(self):
        return ("a",0)

    def reward(self, x, a):
        if (x == ("a",self.horizon-1) and a == self.opt_a[x[1]]) or (x == ("b",self.horizon-1) and a == self.opt_b[x[1]]):
            return 1
        return 0

if __name__=='__main__':
    E = Environment()
    rewards = [0,0]
    counts = [0,0]
    for t in range(1000):
        x = E.start_episode()
        while x != None:
            actions = E.get_actions()
            a = np.random.choice(actions)
            (x,r) = E.act(a)
            rewards[a] += r
            counts[a] += 1
    for a in [0,1]:
        assert (np.abs(np.float(rewards[a])/counts[a] -E.reward_dists[a]) < 0.1)


    E = CombinationLock(horizon=3)
    print (E.opt)
    for t in range(10):
        x = E.start_episode()
        while x != None:
            actions = E.get_actions()
            a = np.random.choice(actions)
            old = x
            (x,r) = E.act(a)
            print(old, a, r, x)

    E = StochasticCombinationLock(horizon=3, swap=0.5)
    print (E.opt_a)
    print (E.opt_b)
    for t in range(10):
        x = E.start_episode()
        while x != None:
            actions = E.get_actions()
            a = np.random.choice(actions)
            old = x
            (x,r) = E.act(a)
            print(old, a, r, x)
