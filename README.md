# Frozen-Lake-DQN
 Using this project to learn more about Reinforcement Learning both in theory and practice. 
 - So far I know SARSA, Q-Learning and Bellman equations work in theory but I need pratical experience on how these algorithims work
 - Using youtube tutorial [here](https://www.youtube.com/watch?v=EUrWGTCGzlA) for refernce implementation
    - Noticed that the bot (annoyingly) keeps walking into the same puddle many times during training even though it should've learned not to from the first time it walked into that puddle.
  - So made a custom algorithim where bot learns from it's mistake (negative feedback?) and doesn't repeat the same mistake twice
 - The algorithim is also split into 2 phases: planning/execution and improvisation/learning, trying to mimic human behavior:
     - During Planning phase the bot has a virtual representation of the game it has in it's "mind" and tries to pick a path to explore based on which path is explored the most. Reason being it's closer to knowing whether that path leads to a reward or not.
     - Bot then follows that pre-planned path and start's improvising once it hits uncharted territory.
     - Right now the bot just selects a random action once it hit unknown states but the maybe in the future the bot can use it uses prior knowledge to traverse unknown states. This is a vague idea, hopefully can explore in future
     - The bot also updates it's internal representation of the world with every action it takes. So that during the planning stage it knows which states to avoid and which ones to explore
  - Although the implementation of this alorigthim is specific to this problem. I believe it has some aspects generally applicable to RL problems
  
 - TODO: Prettier format
 - TODO: Have numbers to prove my algorithim is more efficent in this case
  For 500 runs: 
   Numbers from testing custom implementation:
   Average steps taken to reach reward:517.518 and variance:64433.51670941884
   Average path lenght is 27.352 and variance:85.78767134268537

   Average steps taken to reach reward:506.248 and variance:56702.6397755511
   Average path lenght is 26.688 and variance:77.8102765531062

After 5000 runs (took ~5 min):
   Average steps taken to reach reward:504.9858 and variance:61387.775553470696
  Average path lenght is 26.9962 and variance:81.39286413282656

   Note: The alogirthim isn't perfect, a big bottleneck right now is path planning. It tends to plan for the longest path, which is not a good idea in hindsight, as it takes really long to plan if you know a lot of states. Maybe can use attention mechanism to improve this? This is a problem right now because I need to exit the simulation if planning takes too long.


 Numbers from reference implementation (took ~1hr):
 On average it took 7188.45 steps to reach the reward. Variance: 18578403.34090909
 
 On average it took 6700.288 steps to reach the reward. Variance: 19929425.801816363
