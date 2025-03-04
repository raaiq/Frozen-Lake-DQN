# Frozen-Lake-DQN

## Purpose
[//]: <> (What's the purpose of this project?)
The main goal of this project is to learn more about Reinforcement Learning alrogithims, how they work and if possbile improve upon them. In this case, I'm programming an agent for the Frozen Lake environment part of Open AI's Gymnasium. The goal is to traverse from one end of a frozen "lake" to find the reward, without falling into a ice hole.
 
## Background
[//]: <> (Main concepts surrounding this project, i.e Bellman equations, Q-Learning alogrithim, Nueral Networks)


## Reference Design
[//]: <> (Describe the compoenets of the reference implementation and how each implementation works and it's purpose)


## Results and limitations
[//]: <> (Show what were the bottlenecks of the reference implementation and how those can be addressed.)

| No. of Runs | Average no. of steps to reward | Std Dev. |
| -------- | ------- | --------- |
| 100 | 7188.45 |  4310.26 |
| 5000 | 6700.29 | 4464.24 |

- It took about 1 hour to complete 5000 runs
  
The figures above are just  for reaching the reward, not fot the bot learning the path towards the reward. (Need to explore on how much they difference, i.e how many more runs after the bot reaches the reward the first time that it learns the proper path?)
  1. Bot kept falling into the same ice hole multiple times, i.e kept repeating the same mistakes
  2. Bot not learning fast enoguh. i.e the game has 64 possible states, with 4 actions possible for each. So it should take at most 256 tries to get a path to the reward plus some margin left for sceneroies when the bot falls into a ice hole and has to go over known states again.

## Improved Design
[//]: <> (Describe the improved design and how it's better than the reference. Maybe also describe each compoenent and their purpose maybe even future plans for them?)
Tried to solve the problem by first giving the bot negative reward (in addition to it's existing positive reward) when it fell into a ice hole. This approach didn't work as it somehow hindered the agent from learning the path to the reward after the typical 1000 episodes. My guess is, it has to do with the limited expressiveness of the NN due to it's small size (only 1 hidden layer and 16 nuerons in it). It's just a conjecture, should confirm. So I decided to make the implmentation using Q-tables but I admittedly don't understand how Q-tables work very well so I made a custom algorithim tailored towards this specific problem. 

In an attempt to mimic human thinking, the algorithim is split into 2 phases: plannig phase and exploration/learning phase. During the planning phase, the bot picks a path to explore which isn't fully explored yet. Then, the bot navigates that path, and starts exploring it's enviroment once it hits uncharted territory. The agent also has an internal representation of it's enviroment, which it updates after every step. The bot also keeps track of the path it has taken so far from the start of the game, so that when it reaches the reward, it already knows it's path. This isn't a perfect solution but it works. When the bot falls into a ice hole, it updates it's internal map to avoid that state or any state which strictly leads to that ice hole.

## Results and limitations of new design
[//]: <> (Shows numbers to back-up that your design it better)

 | No. of Runs | Average no. of steps to reward | Steps Std Dev. | Mean Path Lenght | Path Std Dev. |
 | -------- | ------- | --------- | ----- | ----- |
 | 500      | 517.52  |  253.84   | 27.35 | 9.26  |
 | 5000     | 504.99  | 238.12    | 27    | 9.01  |
- It took about 5 min to complete 5000 runs

- Path planning is inefficent
- Path to reward is sub-optimal as well. Bot goes backwards, then forwards, repeated states in path

## Future Plans
[//]: <> (Shows numbers to back-up that your design it better)
1. For the reference implementation, explore how many more runs are needed for the bot to learn the path to reward 
2. Implement the negative feedback algorithim using Q-Tables
3. From testing ,the custom algorithim tends to pick the longest path during planning stage. Confirm if true. If it is then make it not be the case, and be more efficent
4. Implement the negative feedback algorithim using nueral networks



 - Using youtube tutorial [here](https://www.youtube.com/watch?v=EUrWGTCGzlA) for refernce implementation
 - The algorithim is also split into 2 phases: planning/execution and improvisation/learning, trying to mimic human behavior:
     - During Planning phase the bot has a virtual representation of the game it has in it's "mind" and tries to pick a path to explore based on which path is explored the most. Reason being it's closer to knowing whether that path leads to a reward or not.
     - Bot then follows that pre-planned path and start's improvising once it hits uncharted territory.
     - Right now the bot just selects a random action once it hit unknown states but the maybe in the future the bot can use it uses prior knowledge to traverse unknown states. This is a vague idea, hopefully can explore in future
     - The bot also updates it's internal representation of the world with every action it takes. So that during the planning stage it knows which states to avoid and which ones to explore
  - Although the implementation of this alorigthim is specific to this problem. I believe it has some aspects generally applicable to RL problems
  
 - TODO: Prettier format
  
   Note: My alogirthim isn't perfect, a big bottleneck right now is path planning. It tends to plan for the longest path, which is not a good idea in hindsight, as it takes really long to plan if you know a lot of states. Maybe can use attention mechanism to improve this? This is a problem right now because I need to exit the simulation if planning takes too long.


 Numbers from reference implementation (took ~1hr):
 On average it took 7188.45 steps to reach the reward. Variance: 18578403.34090909
 
 On average it took 6700.288 steps to reach the reward. Variance: 19929425.801816363
