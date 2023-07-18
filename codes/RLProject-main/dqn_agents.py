from inspect import stack
from base_agent import RLAgent
from collections import deque
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, LeakyReLU
# from tensorflow.keras.optimizers import Adam

class DQNAgent(RLAgent):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=30000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.model = self._build()

    def _build(self):
        # CNN for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(8,8), strides = 4, activation=LeakyReLU(), input_shape=self.state_size))
        model.add(Conv2D(64, kernel_size=(4,4), strides = 2, activation=LeakyReLU()))
        model.add(Conv2D(64, (3,3), activation=LeakyReLU()))
        model.add(Flatten())
        model.add(Dense(512, activation=LeakyReLU()))
        model.add(Dense(self.action_size, kernel_initializer="uniform", activation="linear"))

        model.compile(loss="mse", optimizer='adam', metrics=["accuracy"])
        return model

    def take_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.expand_dims(state, 0))
        return np.argmax(act_values[0])  # returns action

    def train(self, train_batch_size):
        if len(self.memory) < train_batch_size:
            return
        minibatch = random.sample(self.memory, train_batch_size)

        states, _, rewards, next_states, dones = zip(*minibatch)
        states = np.stack(states)
        rewards = np.stack(rewards)
        next_states = np.stack(next_states)
        dones = np.stack(dones)
        target = rewards + (1 - dones) * self.gamma * np.amax(self.model.predict(next_states), axis=-1)
        target_f = self.model.predict(states)
        for i, (_, action, _, _, _) in enumerate(minibatch):
            target_f[i][action] = target[i]
        self.model.fit(states, target_f)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class DQN_PER_Agent(DQNAgent):

    def memorize(self, state, action, reward, next_state, done):
        target_q = reward + (1 - done) * self.gamma * np.amax(self.model.predict(np.expand_dims(next_state, 0))[0])
        current_q = self.model.predict(np.expand_dims(state, 0))[0]
        mse = (current_q[action] - target_q)**2
        if mse > 1 or done or ((current_q.max() - current_q.min())**2 < 1):
            self.memory.append((state, action, reward, next_state, done))
