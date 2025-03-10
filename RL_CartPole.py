import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = gym.make("CartPole-v1")
is_ipython ='inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()
device =torch.device(
    "cuda" if torch.cuda.is_available() else 
    "mps" if torch.backends.mps.is_available() else 
    "cpu"
)

class DQN(nn.Module):
    def __init__(self,n_oberservations, n_actions):
        super(DQN,self).__init__()
        self.layer1= nn.Linear(n_oberservations,128)
        self.layer2= nn.Linear(128,128)
        self.layer3 = nn.Linear(128,n_actions)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x= F.relu(self.layer2(x))
        return self.layer3(x)
    
Transition = namedtuple("Transition",
                        ("state","action","next_state","reward"))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen= capacity)
    
    def  push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.load_state_dict())

optimizer = optim.AdamW(policy_net.parameters(),lr=LR,amsgrad=True)
memory = ReplayMemory(100)

steps_done=0

def select_action(action):
    global steps_done
    sample = random.random()
    eps_threshld = EPS_END + (EPS_START -EPS_END) / math.exp(steps_done/EPS_DECAY)
    steps_done +=1
    if sample > eps_threshld:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1,1)
    else:
        return torch.tensor([[env.action_space.sample()]],device=device,dtype=torch.long)

episode_durations =[]

def plot_durations(show_result=False):
    plt.figure()
    durations_t = torch.tensor(episode_durations, dtype= torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())
    if len(durations_t) >=100:
        means = durations_t.unfold(0,100,1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99),means))
        plt.plot(means.numpy())
    plt.pause(0.001)
    if is_ipython:
        display.display(plt.gcf())
        display.clear_output(wait=True)
    else:
        display.display(plt.gcf())
        
