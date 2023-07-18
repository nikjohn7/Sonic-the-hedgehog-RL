import os, json, shutil
from typing import Dict, Type
from absl import app, flags
import retro
import numpy as np
from skimage.transform import resize
from base_agent import RLAgent
from dqn_agents import DQNAgent, DQN_PER_Agent
from ddqn_agents import DDQNAgent


os.environ['CUDA_VISIBLE_DEVICES'] = "2"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


FLAGS = flags.FLAGS

flags.DEFINE_string('name', None, 'The folder name used for saving the model and the logs. None defaults to the class name of the agent')
flags.DEFINE_string('env', 'sonic1', 'The environment to use')
flags.DEFINE_string('scenario', 'contest', 'The scenario to use from the environment')
flags.DEFINE_string('agent', 'dqn', 'The model to use')
flags.DEFINE_integer('episodes', 100, 'Number of episodes')
flags.DEFINE_integer('max_steps_per_episode', 5000, 'Max number of steps for an episode (can be None for infinite steps)')
flags.DEFINE_list('resized_state_size', [128, 128], 'Resize the state as image to this')
flags.DEFINE_bool('continue_training', False, 'Set False to force a fresh training')
flags.DEFINE_integer('train_every_n_steps', 64, 'Steps after which to execute train')
flags.DEFINE_integer('train_batch_size', 64, 'Batch size to use for training')


ENV_DICT: Dict[str, str] = {
    'sonic1': 'SonicTheHedgehog-Genesis',
    }
MODEL_DICT: Dict[str, Type[RLAgent]] = {
    'dqn': DQNAgent,
    'dqn_per': DQN_PER_Agent,
    'ddq': DDQNAgent
    }
ACTION_WRAPPER = {
    'sonic1' : {
        # No Operation
        0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # Left
        1: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        # Right
        2: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        # Left, Down
        3: [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        # Right, Down
        4: [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        # Down
        5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        # Down, B
        6: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        # B
        7: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
}


def mod_reward(reward, done, info):
    if (info['prev_lives'] > info['lives']) and done:
        return -1000
    return reward - 1.


def train(
    env: retro.RetroEnv,
    agent: RLAgent,
    output_dir,
    episodes,
    max_steps_per_episode = None,
    resized_state_size = [128, 128],
    action_wrapper = None,
    continue_training = True,
    train_every_n_steps = 64,
    train_batch_size = 64):

    if os.path.exists(output_dir):
        if continue_training:
            agent.load(os.path.join(output_dir, 'saved_models'))
        else:
            shutil.rmtree(output_dir)

    os.makedirs(os.path.join(output_dir, 'logs'))
    os.makedirs(os.path.join(output_dir, 'saved_models'))
    os.makedirs(os.path.join(output_dir, 'best_model'))
    episode_rewards_info = {}

    train_step = 0
    total_rewards = []
    best_reward = 0.
    for episode in range(episodes):
        state = env.reset()
        if resized_state_size:
            state = resize(state, resized_state_size)
        done = False
        total_reward = 0.
        step = 0
        step_rewards = []
        while (not max_steps_per_episode) or (step < max_steps_per_episode):
            action = agent.take_action(state)
            buttons = action_wrapper[action] if action_wrapper else action
            next_state, reward, done, info = env.step(buttons)
            if resized_state_size:
                next_state = resize(next_state, resized_state_size)

            # Modify rewards here if required.
            total_reward += reward
            # reward = mod_reward(reward, done, info)
            step_rewards.append(reward)
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            step += 1
            train_step += 1
            if train_step % train_every_n_steps == 0:
                agent.train(train_batch_size)
                agent.save(os.path.join(output_dir, 'saved_models'))
            if done:
                break

        total_rewards.append(total_reward)
        print("episode: {}/{}, score: {}, e: {:.2}, done: {}"
                      .format(episode, episodes, total_reward, agent.epsilon, done))
        if total_reward > best_reward:
            agent.save(os.path.join(output_dir, 'best_model'))
            best_reward = total_reward
        episode_rewards_info['total_rewards'] = total_rewards
        episode_rewards_info['episode_{}'.format(episode)] = step_rewards
        with open(os.path.join(output_dir, 'logs', 'rewards.json'), 'w') as f:
            json.dump(episode_rewards_info, f)


def main(_):
    env_id = ENV_DICT[FLAGS.env]
    env = retro.make(env_id, scenario=FLAGS.scenario)
    channels = env.observation_space.shape[-1]

    state_space = FLAGS.resized_state_size + [channels] if FLAGS.resized_state_size else list(env.observation_space.shape)
    action_space = len(ACTION_WRAPPER[FLAGS.env]) if FLAGS.env in ACTION_WRAPPER else env.action_space.n
    Agent_Class = MODEL_DICT[FLAGS.agent]
    agent = Agent_Class(state_space, action_space)
    agent_name = FLAGS.name if FLAGS.name else Agent_Class.__name__
    output_dir = os.path.join('outputs', env_id, agent_name)
    env.close()
    env = retro.make(env_id, scenario=FLAGS.scenario, record=output_dir)
    train(env,
          agent,
          output_dir,
          FLAGS.episodes,
          FLAGS.max_steps_per_episode,
          FLAGS.resized_state_size,
          ACTION_WRAPPER[FLAGS.env] if FLAGS.env in ACTION_WRAPPER else None,
          FLAGS.continue_training,
          FLAGS.train_every_n_steps,
          FLAGS.train_batch_size)
    env.close()


if __name__ == '__main__':
    app.run(main)
