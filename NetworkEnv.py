import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
from sklearn.preprocessing import MinMaxScaler

from environmental_variables import NUMBER_OF_HOSTS, NUMBER_OF_PATHS

from NetworkEngine import NetworkEngine


class NetworkEnv(Env):

    def __init__(self, engine: NetworkEngine):
        self.requests = 0
        r = int(np.random.normal(24, 8))
        while r > 32 or r < 1:
            r = int(np.random.normal(24, 8))
        self.max_requests = r
        self.done = False

        self.engine = engine

        self.observation_space = Box(
            low=np.zeros((NUMBER_OF_HOSTS, NUMBER_OF_HOSTS, NUMBER_OF_PATHS, 1), dtype=np.float32),
            high=np.full((NUMBER_OF_HOSTS, NUMBER_OF_HOSTS, NUMBER_OF_PATHS, 1), 100, dtype=np.float32),
            dtype=np.float32)

        self.action_space = Discrete(NUMBER_OF_PATHS)
        self.state = np.full((NUMBER_OF_HOSTS, NUMBER_OF_HOSTS, NUMBER_OF_PATHS, 1), 100, dtype=np.float32)

    def step(self, actions: dict):

        for host, dsts in actions.items():
            self.engine.set_active_path(host, dsts)

        self.engine.simmulate_turn()
        self.requests += 1

        self.state = None

        self.done = self.engine.communication_done()

        rewards = []        
        for host in self.engine.get_all_hosts():
            reward = 0
            bw = self.engine.components[host].get_neighbors_bw()
            #print("\n bw reward: ", bw)
            if bw > 75:
                reward += 20  #50 #50 #50 #20
            elif bw > 50:
                reward += 50  #30 #30 #30 #50
            elif bw > 25:
                reward += 20  #pass #pass #10 #20
            elif bw > 0:
                reward -= 60  #-20 #70 #40 #60
            else:
                reward -= 120 #70 #100 # 100 #120
            rewards.append(reward)

        states = {}

        for host in self.engine.get_all_hosts():
            states[host] = self.engine.get_state(host, 3)

        return states, rewards, self.done, {}

    def render(self):
        pass

    def get_state(self):
        return self.state

    def reset(self, new_tm=False):
        self.done = False
        self.engine.reset(new_tm)
