# In Progress/Future Work

This folder consists of a few of our experiments that either failed or couldn't be trained in time, but is still worth taking a look at. We hope to work beyond the deadline of this project, and try and work on these for better understanding of core RL functioning.

## Files

`agents.py`: Consists of all agents that are being implemented. Currently, it has `DQN` and `DQN_PER` agents. New agents can be added with ease

`models.py`: The neural network structure for each model. Currently this only has a single model structure. Dueling DQN, Rainbow DQN etc can be added

`main.py`: The main file that is called to train an agent. 

`helpers.py`: utility functions

`use_per.py`: A custom attempt at implementing Prioritized Experience Replay using Binary trees

`custom_reward.py`: An attempt at creating a custom reward function to help the agent prioritise collecting coins along with just finishing the level


## Usage

- Add your agent to the `agents.py` file. 

- Then in the `train.py`, import and call them
