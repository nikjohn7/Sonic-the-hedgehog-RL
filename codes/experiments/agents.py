import math

import numpy as np
import random

from use_per import PER_History

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Add, Lambda, MaxPooling2D, GaussianDropout
from tensorflow.keras.models import Sequential, Model, load_model

from collections import deque


import numpy as np
import retro


class DQN_Agent:

    def __init__(self, input_shape, action_space):

        self.input_shape = input_shape
        self.action_space = action_space

        self.gamma = 0.96
        self.model_lr = 0.0001
        self.target_lr = 0.0001

        self.obs_timesteps = 5000

        self.epsilon = 1.0
        self.initial_epsilon = 1.0
        self.end_epsilon = 0.01
        

        self.exploration_timesteps = 4000000
        self.frame_per_action = 4
        self.mod_target = 10000 
        

        self.add_noise = False


        self.train_timestep = 2000 
        
        self.batch_size = 128

        self.history = deque(maxlen=50000)
        
        self.model = None 

        self.target_network = None

        self.idx_to_act = action_switch = {
            0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #Nothing
            1: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], #Left
            2: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], #Right
            3: [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], #Left down
            4: [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], #Right down
            5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], #Down
            6: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], #Down + B
            7: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #B
        }
    
    
    def get_action(self, observation):

        if (np.random.rand() <= self.epsilon) and (not self.add_noise):
            action_num = random.randrange(self.action_space)
        else:

            predicted_actions = self.model.predict(observation)
            action_num = np.argmax(predicted_actions)
            
        chosen_action = self.idx_to_act[action_num]

        return action_num, chosen_action
    
    
    def save_history(self, state, act_num, reward, next_state, done):
        self.history.append((state, act_num, reward, next_state, done))

        
    def modify_target_network(self):
        self.target_network.set_weights(self.model.get_weights())
        print("Target Network Modified")

        
    def replay(self):

        sampled_batch = random.sample(self.history, self.batch_size)
        
        new_input = np.zeros(((self.batch_size,) + self.input_shape)) 
        new_target = np.zeros(((self.batch_size,) + self.input_shape)) 
        action, reward, completed = [], [], []

        for i in range(self.batch_size):
            new_input[i,:,:,:] = sampled_batch[i][0]
            action.append(sampled_batch[i][1])
            reward.append(sampled_batch[i][2])
            new_target[i,:,:,:] = sampled_batch[i][3]
            completed.append(sampled_batch[i][4])

        prediction = self.model.predict(new_input) 
        
        optimal_pred = self.model.predict(new_target)
        target_pred = self.target_network.predict(new_target)

        for sample in range(self.batch_size):

            if completed[sample]:
                score = reward[sample]
            else:
 
                chosen_action = np.argmax(optimal_pred[sample])
                score = reward[sample] + self.gamma * (target_pred[sample][chosen_action])
        
        prediction[sample][action[sample]] = score

        loss = self.model.train_on_batch(new_input, prediction)

        return np.amax(prediction[-1]), loss


    def load_models(self, model_path, target_path):

        self.model = load_model(model_path)
        self.target_network = load_model(target_path)
        print("Models Loaded")

    def save_models(self, model_path, target_path):
        self.model.save(model_path)
        self.target_network.save(target_path)

        print("Models saved")
        
        
class DQN_PER_Agent(DQN_Agent):
    
    def __init__(self, input_shape, action_space):
        DQN_Agent.__init__(self, input_shape, action_space)
        self.obs_timesteps = 10000
        self.history = PER_History(5000)
        
    def save_history(self, state, act_num, reward, next_state, done):
        exp = (state, act_num, reward, next_state, done)
        
        _, _, error = self.compute_targets([exp])

        self.history.add(exp, error[0])
        
        
    def compute_targets(self, mini_batch):
        
        batch_size = len(mini_batch)

        update_input = np.zeros(((batch_size,) + self.input_shape)) 
        update_target = np.zeros(((batch_size,) + self.input_shape)) 
        action, reward, done = [], [], []

        # Extract information from the sampled memories.
        for i in range(batch_size):
            # Append each state (1, 128, 128, 4) to update_input
            update_input[i,:,:,:] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            # Append Next State (1, 128, 128, 4) to update_target
            update_target[i,:,:,:] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        prediction = self.model.predict(update_input) 
        
        # Predict the values associated with acting optimally from next state.
        next_state_pred = self.model.predict(update_target)
        target_pred = self.target_network.predict(update_target)

        error = []
        for sample_idx in range(batch_size):
            # Q-value prior to update
            prev_q = next_state_pred[sample_idx][action[sample_idx]]

            # For terminal actions, set the value associated with the action to
            # the observed reward
            if done[sample_idx]:
                predicted_val = reward[sample_idx]
            else:
                next_action = np.argmax(next_state_pred[sample_idx])
                predicted_val = reward[sample_idx] + self.gamma * (target_pred[sample_idx][next_action])
            error.append(abs(prev_q - predicted_val))
        
        # Update the q value associated with the action taken.
        prediction[sample_idx][action[sample_idx]] = predicted_val

        return update_input, prediction, error
    
    def replay(self):
        
        sampled_batch, indices = self.history.sample(self.batch_size)
        
        new_input, prediction, error = self.compute_targets(sampled_batch)

        for i in range(self.batch_size):
            ind = indices[i]
            self.history.update(ind, error[i])

        loss = self.model.train_on_batch(new_input, prediction)

        return np.amax(prediction[-1]), loss
        
    
        