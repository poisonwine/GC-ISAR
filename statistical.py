import os
import numpy as np
from rl_modules.replay_buffer import replay_buffer
from her_modules.her import her_sampler
from train import get_args
import gym
import pickle

def compute_reward(achieved_goal, goal, info):
    # Compute distance between goal and the achieved goal.
    d = goal_distance(achieved_goal, goal)
    return -(d > 0.01).astype(np.float32)

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            'action_space': env.action_space
            }
    params['max_timesteps'] = env._max_episode_steps
    return params


def get_full_envname(name):
    dic = {
        'FetchReach':'FetchReach-v1',
        'FetchPush': 'FetchPush-v1',
        'FetchSlide': 'FetchSlide-v1',
        'FetchPick': 'FetchPickAndPlace-v1',
        'HandReach':'HandReach-v0',
    }
    if name in dic.keys():
        return dic[name]
    else:
        return name

def statistical(env_name):
    load_path_expert = f'./offline_data/expert/{env_name}/'
    load_path_random = f'./offline_data/random/{env_name}/'
    buffer_name = 'buffer'
    path_expert = os.path.join(load_path_expert, f'{buffer_name}.pkl')
    path_random = os.path.join(load_path_random, f'{buffer_name}.pkl')
    with open(path_expert, "rb") as fp_expert:  
        data_expert = pickle.load(fp_expert)   
        size_expert = data_expert['o'].shape[0]
        # shapes = data_expert['ag'].shape
        # print(shapes)
        shapes = data_expert['g'].shape
        print(data_expert['o'].shape, data_expert['u'].shape,data_expert['g'].shape)
        ags = data_expert['ag'][:,:-1,:].reshape(shapes[0]*shapes[1],shapes[2])
        gs =  data_expert['g'].reshape(shapes[0]*shapes[1],shapes[2])
        final_ags = data_expert['ag'][:,-1,:].reshape(shapes[0], -1)
        final_dgs = data_expert['g'][:,-1,:].reshape(shapes[0], -1)

        final_distance = np.sum(np.linalg.norm(final_ags-final_dgs)) / shapes[0]
        print('final_distance', final_distance)
        rewards = compute_reward(ags, gs, info=None) + 1.0
        traj_r = rewards.reshape(shapes[0], -1).sum(axis=-1)
        print(traj_r.shape)
        avg_rewards = rewards.sum() / shapes[0]
        print('avg reward',avg_rewards,'max_expert reward',np.max(traj_r, axis=-1),'min_expert reward',np.min(traj_r, axis=-1))


    # with open(path_random, "rb") as fp_random:
    #     data_random = pickle.load(fp_random)
    #     size_random = data_random['o'].shape[0]   
    #     shapes = data_random['g'].shape
    #     # print(shape2)
    #     ags = data_random['ag'][:,:-1,:].reshape(shapes[0]*shapes[1],shapes[2])
    #     gs =  data_random['g'].reshape(shapes[0]*shapes[1],shapes[2])
    #     rewards = compute_reward(ags, gs, info=None) + 1.0
    #     traj_r = rewards.reshape(shapes[0], -1).sum(axis=-1)
    #     print(traj_r.shape)
    #     avg_rewards = rewards.sum() / shapes[0]
    #     print('avg reward random',avg_rewards,'max expert reward',np.max(traj_r, axis=-1),'min_expert reward',np.min(traj_r, axis=-1))

statistical('FetchPush')