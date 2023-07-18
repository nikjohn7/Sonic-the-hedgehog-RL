# Sonic the Hedgehog using Deep Q-Learning

The code is provided in two major classes:

- dqn_agents.py (The code for the agent)
- train.py (Used for training the agent)

The Agents are a subclass of RLAgent defined in base_agent.py that defines a standard interface for the agents.

The required methods to be overriden and exposed in the Agents are:
- _build (To build the Neural Netowrk)
- memorize (To collect the experience)
- train (To train the agent using collected experience)
- take_action (To generate an action for the given state)

The train script creates the environment and the specified agent and executes the training for the specified number of episodes.

The following command can be used to execute the training [Please refer to train.py for the full list of available arguments]:

```shell

python train.py --env=sonic1 --agent=dqn_per --episodes=100 --max_steps_per_episode=5000 --train_bath_size=64
```

The agents can be changed using the ```--agent``` argument. The available values for the argument currently are [dqn,dqn_per,ddqn]

Some incomplete experiments with DDQN+PER and DQN with custom rewards have been placed in the experiments folder.
