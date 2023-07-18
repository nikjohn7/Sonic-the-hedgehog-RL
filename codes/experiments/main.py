import math
import os

from collections import deque
import numpy as np
import random
import retro

from helpers import resize_img,view_img
from models import Models
from agents import DQN_Agent, DQN_PER_Agent

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Add, Lambda, MaxPooling2D, GaussianDropout
from tensorflow.keras.models import Sequential, Model, load_model

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

NUM_EPISODES = 5000

LOAD_MODEL = False
PER_AGENT = True
DUELING = False
EPSILON = 0


env = retro.make(game="SonicTheHedgehog-Genesis", state="GreenHillZone.Act1", scenario="contest")
    
img_rows = 128
img_cols = 128          
img_stack = 4        

action_space = 8         # 8 valid button combinations

input_shape = (img_rows, img_cols, img_stack)

stat_path = '../statistics/dqn'
model_path = '../models/dqn'

if PER_AGENT:
    print('PER agent')
    stat_path += '_PER'
    model_path+= '_PER'
    dqn_agent = DQN_PER_Agent(input_shape, action_space)
else:
    dqn_agent = DQN_Agent(input_shape, action_space)



dqn_agent.model = Models.dqn(input_shape, action_space, dqn_agent.model_lr)
dqn_agent.target_network = Models.dqn(input_shape, action_space, dqn_agent.target_lr)

stat_path += '_stats.csv'
model_path = model_path + '_main.h5'
target_network_path = model_path + '_target.h5'


if (LOAD_MODEL):
    dqn_agent.load_models(model_path, target_network_path)

# Modify statrting epsilon value
if (EPSILON == 0):
    dqn_agent.epsilon = dqn_agent.initial_epsilon
elif (EPSILON == 1):
    dqn_agent.epsilon = ((dqn_agent.initial_epsilon - dqn_agent.final_epsilon) / 2)
else:
    dqn_agent.epsilon = dqn_agent.final_epsilon


total_timestep = 0              # Total number of timesteps over all episodes.
for episode in range(NUM_EPISODES):
    completed = False
    cum_reward = 0          
    timestep = 0
    currloss=0
    init_state  = env.reset()


    experience = resize_img(init_state, size=(img_rows, img_cols))

    exp_stack = np.stack(([experience]*img_stack), axis = 2)

    exp_stack = np.expand_dims(exp_stack, axis=0) # 1x64x64x4

    # Punish the agent for not moving forward
    prev_state = {}
    immobile_count = 0


    while not completed:

        act_num, action = dqn_agent.get_action(exp_stack)
        obs, reward, completed, info = env.step(action)
        # env.render()

        if (prev_state == info):
            immobile_count += 1
        else:
            immobile_count = 0
        prev_state = info

        cum_reward += reward      
        if (immobile_count > 20):
            reward -= 1

        timestep += 1
        total_timestep += 1

        obs = resize_img(obs, size=(img_rows, img_cols))


        obs = np.reshape(obs, (1, img_rows, img_cols, 1))

        new_exp_stack = np.append(obs, exp_stack[:, :, :, :3], axis=3)

        dqn_agent.save_history(exp_stack, act_num, reward, new_exp_stack, completed)
        exp_stack = new_exp_stack

        if (total_timestep >= dqn_agent.obs_timesteps):

            if ((total_timestep % dqn_agent.mod_target) == 0):
                dqn_agent.modify_target_network()

            if ((total_timestep % dqn_agent.train_timestep) == 0):
                pp, currloss = dqn_agent.replay()
                dqn_agent.save_models(model_path, target_network_path)

            if (dqn_agent.epsilon > dqn_agent.end_epsilon):

                dec = ((dqn_agent.initial_epsilon - dqn_agent.end_epsilon) / dqn_agent.exploration_timesteps)
                dqn_agent.epsilon -= dec

        # print(info)
        if timestep%100==0:
            print("Epsisode:", episode, " Timestep:", timestep, " Action:", act_num, " Episode Reward Sum:", cum_reward, "  Epsilon:", dqn_agent.epsilon, " loss:", currloss)

    # Save mean episode reward at the end of the episode - append to stats file            
    with open(stat_path, "a") as stats_fd:
        reward_str = "Epsiode Cummulative Reward: " + str(cum_reward) + ", Episode Timesteps: " +  str(timestep) + ",\n"
        stats_fd.write(str(reward_str))
env.close()