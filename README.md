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
 
