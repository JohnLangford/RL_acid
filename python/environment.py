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
        return self.make_obs(self.state)

    def get_actions(self):
        if self.state is None:
            raise Exception("Episode not started")
        if self.h == self.horizon:
            return None
        return self.actions
    
    def make_obs(self, s):
        return s

    def act(self, a):
        if self.state is None:
            raise Exception("Episode not started")
        r = np.random.binomial(1, self.reward(self.state,a))
        self.h += 1
        if self.h == self.horizon:
            self.state = None
        else:
            self.state = self.transition(self.state,a)
        return(self.make_obs(self.state), r)

    def get_num_actions(self):
        return len(self.actions)

    def is_tabular(self):
        return True

    def get_dimension(self):
        assert not self.is_tabular(), "Not a featurized environment"
        return self.dim

class CombinationLock(Environment):
    def __init__(self, horizon=5, dim=None):
        self.horizon=horizon
        self.actions = [0,1]
        self.opt = np.random.choice(self.actions,size=self.horizon)
        if dim is None:
            self.dim = None
        else:
            self.dim = 2*horizon+dim

    def transition(self, x, a):
        if x is None:
            raise Exception("Not in any state")
        if x[0] == 1 and a==self.opt[x[1]]:
            return [1,x[1]+1]
        return [0,x[1]+1]
            
    def make_obs(self,x):
        if x is None:
            return x
        if self.dim == None:
            return x
        else:
            v = np.zeros(self.dim,dtype=int)
            v[2*self.horizon:] = np.random.binomial(1,0.5, self.dim-2*self.horizon)
            v[2*x[1]+x[0]] = 1
            return (v)

    def start(self):
        return [1,0]

    def reward(self, x, a):
        if x == [1,self.horizon-1] and a == self.opt[x[1]]:
            return np.random.binomial(1, 0.5)
        return 0

    def is_tabular(self):
        return (self.dim == None)

class StochasticCombinationLock(Environment):
    def __init__(self, horizon=5, swap=0.1, dim=None):
        self.horizon=horizon
        self.swap = swap
        self.actions = [0,1]
        self.opt_a = np.random.choice(self.actions,size=self.horizon)
        self.opt_b = np.random.choice(self.actions,size=self.horizon)
        if dim is None:
            self.dim = None
        else:
            self.dim = 3*self.horizon+dim

    def transition(self,x,a):
        if x is None:
            raise Exception("Not in any state")
        b = np.random.binomial(1,self.swap)
        if x[0] == 0 and a == self.opt_a[x[1]]:
            if b == 0:
                return([0, x[1]+1])
            else:
                return([1, x[1]+1])
        if x[0] == 1 and a == self.opt_b[x[1]]:
            if b == 0:
                return([1, x[1]+1])
            else:
                return([0, x[1]+1])
        else:
            return([2, x[1]+1])

    def make_obs(self,x):
        if x is None or self.dim == None:
            return x
        else:
            v = np.zeros(self.dim,dtype=int)
            v[3*self.horizon:] = np.random.binomial(1, 0.5, self.dim-3*self.horizon)
            v[3*x[1]+x[0]] = 1
            return (v)

    def start(self):
        return [0,0]

    def reward(self, x, a):
        if (x == [0,self.horizon-1] and a == self.opt_a[x[1]]) or (x == [1,self.horizon-1] and a == self.opt_b[x[1]]):
            return np.random.binomial(1, 0.5)
        return 0

    def is_tabular(self):
        return (self.dim == None)

class RandomGridWorld(Environment):
    """
    A M x M grid with walls and a trembling hand. 
    Horizon is always 2 M
    """
    def __init__(self, M, swap=0.1, dim=2, noise=0.0):
        self.M = M
        self.swap = swap
        self.noise = noise
        self.dim = dim
        self.seed = 147
        np.random.seed(self.seed)
        self.goal = None

        self.maze = self.generate_maze(self.M)
        self.state = None

        self.actions = [(0,1), (0,-1), (1,0), (-1,0)]
        print("ENV: Generated Random Grid World")
        print("Size: %dx%d, Start: [%d,%d], Goal: [%d,%d], H: %d, Cells: %d" % (self.M, self.M, 0, 0, self.goal[1], self.goal[0], self.horizon, np.count_nonzero(self.maze)))
        self.print_maze()

    def is_tabular(self):
        return False

    def generate_maze(self, M):
        """ 
        Adapted from http://code.activestate.com/recipes/578356-random-maze-generator/
        """
        mx = M; my = M
        maze = np.matrix(np.zeros((mx,my)))
        dx = [0, 1, 0, -1]; dy = [-1, 0, 1, 0] # 4 directions to move in the maze
        stack = [(0,0)]

        while len(stack) > 0:
            (cx, cy) = stack[-1]
            maze[cy,cx] = 1
            # find a new cell to add
            nlst = [] # list of available neighbors
            for i in range(4):
                nx = cx + dx[i]; ny = cy + dy[i]
                if nx >= 0 and nx < mx and ny >= 0 and ny < my:
                    if maze[ny,nx] == 0:
                        # of occupied neighbors must be 1
                        ctr = 0
                        for j in range(4):
                            ex = nx + dx[j]; ey = ny + dy[j]
                            if ex >= 0 and ex < mx and ey >= 0 and ey < my:
                                if maze[ey,ex] == 1: ctr += 1
                        if ctr == 1: nlst.append(i)
            # if 1 or more neighbors available then randomly select one and move
            if len(nlst) > 0:
                ir = nlst[np.random.randint(0, len(nlst))]
                cx += dx[ir]; cy += dy[ir]
                stack.append((cx, cy))
            elif self.goal is None:
                self.horizon = 2*len(stack)
                (gx,gy) = stack.pop()
                self.goal = [gx,gy]
            else:
                stack.pop()

        return(maze)

    def make_obs(self,s):
        if s is None:
            return None
        tmp = s.copy()
        v = np.random.normal(0, self.noise, self.dim)
        if self.dim > 2:
            mult = np.random.choice(range(self.M), size=self.dim-2)
            tmp.extend(mult)
        return tmp+v

    def start(self):
        return [0,0]

    def reward(self,x,a):
        if x == self.goal:
            return 1
        return 0

    def transition(self, x, a):
        if x == self.goal:
            return None
        nx = x[0]+a[0]
        ny = x[1]+a[1]
        if nx < 0 or nx >= self.M or ny < 0 or ny >= self.M:
            ## Cannot go off the grid
            return x
        if self.maze[ny,nx] == 0:
            ## Cannot enter a wall
            return x
        else:
            z = np.random.binomial(1, self.swap)
            if z == 1:
                return x
            else:
                return [nx,ny]

    def print_maze(self):
        maze = self.maze
        for i in range(self.M):
            for j in range(self.M):
                if maze[i,j] == 0:
                    print(" ",end="")
                elif self.state is not None and self.state[0]==j and self.state[1]==i:
                    print("A",end="")
                elif self.goal[0]==j and self.goal[1]==i:
                    print("G",end="")
                elif maze[i,j] == 1:
                    print(".",end="")
            print("")


if __name__=='__main__':
    import sys, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', action='store', choices=['MAB', 'combolock', 'stochcombolock', 'maze'])
    Args = parser.parse_args(sys.argv[1:])

    if Args.env == 'MAB':
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

    if Args.env == 'combolock':
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

    if Args.env == 'stochcombolock':
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

    if Args.env == 'maze':
        E = RandomGridWorld(M=3,swap=0.1, dim=2, noise=0.0)
        T = 0
        while True:
            T += 1
            x = E.start_episode()
            if T % 100 == 0:
                print("Iteration t = %d" % (T))
            while x is not None:
                E.print_maze()
                print(x)
                actions = E.get_actions()
                a = np.random.choice(len(actions))
                (x,r) = E.act(actions[a])
                if r == 1:
                    print("Success: T = %d" % (T))
                    sys.exit()
