import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F #What is this?

class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.out = nn.Linear(h1_nodes, out_actions)

    def forward(self, x):
        x= F.relu(self.fc1(x))
        x= self.out(x)
        return x
    
class ReplayMemory():

    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)
    
    def __len__(self):
        return len(self.memory)
    
class FrozenLakeDQL():
    learning_rate_a = 0.001
    discount_factor_g = 0.9
    network_sync_rate = 10
    replay_memory_size = 1000
    mini_batch_size = 32

    loss_fn = nn.MSELoss()
    optimizer = None

    ACTIONS = ['L','D','R','U']

    def train(self, episodes, render= True, is_slippery=False):

        env= gym.make('FrozenLake-v1',map_name="4x4",is_slippery=is_slippery,render_mode='human' if render else None)
        num_states= env.observation_space.n
        num_actions = env.action_space.n

        epsilon = 1
        memory = ReplayMemory(self.replay_memory_size)

        policy_dqn = DQN(num_states,num_states,num_actions)
        target_dqn= DQN(num_states,num_states,num_actions)

        target_dqn.load_state_dict(policy_dqn.state_dict())

        print('Policy (random, before training):')
        self.print_dqn(policy_dqn)
        
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr =self.learning_rate_a)
        rewards_per_episode = np.zeros(episodes)
        epsilon_history = []
        step_count = 0
        for i in range(episodes):
            state = env.reset()[0]
            terminated = False
            truncated = False

            while(not terminated and not truncated):
                if random.random()< epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(state,num_states)).argmax().item()

                new_state,reward,terminated,truncated,_ = env.step(action)
                memory.append((state,action,new_state,reward,terminated))

                state = new_state

                step_count += 1

            if reward ==1:
                rewards_per_episode[i] = reward
            
            if len(memory)> self.mini_batch_size and np.sum(rewards_per_episode) > 0:

                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                epsilon = max(epsilon-1/episodes,0)
                epsilon_history.append(epsilon)

                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count=0  

            
        env.close()
        torch.save(policy_dqn.state_dict(),"frozen_lake_dql.pt")
        plt.figure(1)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)]) # why is this done?
        plt.subplot(121)
        plt.plot(sum_rewards)

        plt.subplot(122)
        plt.plot(epsilon_history)

        plt.savefig('frozen_lake_dql.png')
    

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        num_states = policy_dqn.fc1.in_features

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:

            if terminated:
                target = torch.FloatTensor([reward])
            else:
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount_factor_g* target_dqn(self.state_to_dqn_input(new_state, num_states)).mean()
                    )
            
            current_q = policy_dqn(self.state_to_dqn_input(state, num_states))
            current_q_list.append(current_q)

            target_q = target_dqn(self.state_to_dqn_input(state,num_states))

            target_q[action] = target
            target_q_list.append(target_q)

        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()




    '''
    Converts an state (int) to a tensor representation.
    For example, the FrozenLake 4x4 map has 4x4=16 states numbered from 0 to 15. 

    Parameters: state=1, num_states=16
    Return: tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    '''
    def state_to_dqn_input(self,state:int, num_states:int)->torch.Tensor:
        input_tensor = torch.zeros(num_states)
        input_tensor[state]= 1
        return input_tensor

    def print_dqn(self,dqn):
        num_states= dqn.fc1.in_features

        for s in range(num_states):
            q_values = ''
            for q in dqn(self.state_to_dqn_input(s, num_states)).tolist():
                q_values += "{:+.2f}".format(q)+' '
            q_values = q_values.rstrip()

            best_action = self.ACTIONS[dqn(self.state_to_dqn_input(s, num_states)).argmax()]

            print(f'{s:02},{best_action},{q_values}', end=' ')
            if(s+1)%4==0:
                print()

    def test(self, episodes,is_slippery=False):
        env= gym.make('FrozenLake-v1',map_name="4x4", is_slippery=is_slippery, render_mode="human")
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        policy_dqn.load_state_dict(torch.load("frozen_lake_dql.pt"))
        policy_dqn.eval()

        print('Policy (trained):')
        self.print_dqn(policy_dqn)

        for i in range(episodes):
            state = env.reset()[0]
            terminated = False
            truncated = False
            while(not terminated and not truncated):
                with torch.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(state,num_states)).argmax().item()
                state,reward, terminated,truncated,_ = env.step(action)
            
        env.close()
    
if __name__ == '__main__':
    frozenLake = FrozenLakeDQL()
    is_slippery = False
    # frozenLake.train(1000,False)
    frozenLake.test(10) 