from inspect import stack
from base_agent import RLAgent
from collections import deque
import random
import numpy as np
import copy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, LeakyReLU
# from tensorflow.keras.optimizers import Adam

class DDQNAgent(RLAgent):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = .5  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build()
        self.model2 = self._build()

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
        
#         target_f = self.model.predict(states)
#         target_ff = self.model.predict(next_states)
#         target_ff2 = self.model2.predict(next_states)
        
#         print(target_ff2.shape)
#         print(rewards.shape)
        
#         valid_indexes = np.array(next_states).sum(axis=1) != 0
#         target=np.zeros((len(minibatch), 8))
        target_ = []
        for i, (s, action, r, ns, d) in enumerate(minibatch):
            target = self.model.predict(s[np.newaxis,:])
            target_f1 = self.model.predict(ns[np.newaxis,:])
            target_f2 = self.model2.predict(ns[np.newaxis,:])
            target[0][action] = r + (1 - d) * self.gamma * target_f2[0,:][np.argmax(target_f1[0,:])]
            target_.append(target[0])
        target_f = np.array(target_)
        self.model.fit(states, target_f)
        
        rate = 0.08
        for i, j in zip(self.model2.trainable_variables, self.model.trainable_variables): 
            i.assign(i*(1 - rate)+j*rate)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay