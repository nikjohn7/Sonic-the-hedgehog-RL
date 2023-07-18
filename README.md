# Sonic the Hedgehog Reinforcement Learning

This project involved training reinforcement learning agents to play the classic Sonic the Hedgehog video game using PyTorch and Gym Retro.

## About

The goal was to apply deep reinforcement learning techniques to teach an agent to effectively play through the first Green Hill Zone level in Sonic. This is a complex control task requiring speed, precision, and long-term planning. 

We implemented Deep Q-Learning algorithms including DQN, Prioritized Experience Replay, and Double DQN. The agent receives a reward based on horizontal movement in the level, with additional time bonus. The environment state is the game image pixels.

## Getting Started

The train.py script will train an agent using the chosen algorithm. Evaluation can then be run on a trained model with eval.py.

The requirements are Python 3.6+, Gym Retro, PyTorch, and common packages like Numpy. Use pip to install missing dependencies.

Pretrained models are provided in the /models folder.

## Usage

Train a Double DQN agent:

```
python train.py --model=DoubleDQN  
```

Evaluate a pretrained model:

```
python eval.py --model=DoubleDQN
```

See train.py and eval.py for additional training and evaluation options.

## Performance

Our best Double DQN agent achieved a score over 7000 points. The algorithms were able to learn to effectively control Sonic through difficult obstacles and terrain.

## References

Refer to the full project report for implementation details, additional background, and results analysis.

Let me know if you would like me to expand or modify this README.md draft. The goal was to provide a brief overview of the key information and setup instructions.