import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.base_agent import BaseAgent
from rl_modules.replay_buffer import replay_buffer
from rl_modules.models import actor, critic, value, SacActor
from mpi_utils.normalizer import normalizer
from her_modules.her import her_sampler
import torch.nn as nn
import torch.nn.functional as F
"""
ddpg with HER (MPI-version)

"""
class IQL(BaseAgent):
    def __init__(self, args, env, env_params):
        super().__init__(args, env, env_params) 
  

        # create the network
        self.actor_network = actor(env_params)
        # self.actor_network = SacActor(env_params)
       
        self.critic_network = critic(env_params)
        self.critic_target_network = critic(env_params)

        self.critic_network2 = critic(env_params)
        self.critic_target_network2 = critic(env_params)

        self.v_critic = value(env_params)

        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)
        sync_networks(self.critic_network2)
        sync_networks(self.v_critic)
        # load the weights into the target networks

        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        self.critic_target_network2.load_state_dict(self.critic_network2.state_dict())
        # if use gpu
        if self.args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.critic_target_network.cuda()
            self.critic_network2.cuda()
            self.critic_target_network2.cuda()
            self.v_critic.cuda()

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic1_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        self.critic2_optim = torch.optim.Adam(self.critic_network2.parameters(), lr=self.args.lr_critic)
        self.critic_v_optim =  torch.optim.Adam(self.v_critic.parameters(), lr=self.args.lr_critic)

        self.max_a = env_params['action_max']
        self.max_grad_norm = self.args.max_grad_norm
        self.iql_temperature = self.args.iql_temperature
        self.expectile = self.args.expectile
    


    def _stochastic_actions(self, input_tensor):
        pi = self.actor_network(input_tensor)
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                            size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action
    
    def _deterministic_action(self, input_tensor):
        action = self.actor_network(input_tensor)
        return action

    def _expectile_regression(self, diff: torch.Tensor) -> torch.Tensor:
        weight = torch.where(diff > 0, self.expectile, (1 - self.expectile))
        return weight * (diff**2)

    # update the network
    def _update_network(self, future_p=None):
        # sample the episodes
        sample_batch = self.sample_batch(future_p=future_p)
        transitions = sample_batch['transitions']
        
        # start to do the update
        inputs_norm_tensor, inputs_next_norm_tensor, actions_tensor,r_tensor = self.process_transitions(transitions)
        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()

        # update value network
        with torch.no_grad():
            q_min = torch.min(self.critic_target_network(inputs_norm_tensor, actions_tensor),
                              self.critic_target_network2(inputs_norm_tensor, actions_tensor))
        
        current_v = self.v_critic(inputs_norm_tensor)
        critic_v_loss = self._expectile_regression(q_min - current_v).mean()

        self.critic_v_optim.zero_grad()
        critic_v_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.v_critic.parameters(), self.max_grad_norm)
        sync_grads(self.v_critic)
        self.critic_v_optim.step()

        # update Q critic

        q1 = self.critic_network(inputs_norm_tensor, actions_tensor)
        q2 = self.critic_network2(inputs_norm_tensor, actions_tensor)
        with torch.no_grad():
            next_v = self.v_critic(inputs_next_norm_tensor)
            target_q = r_tensor + self.args.gamma * next_v
        

        value_loss1 = (q1 - target_q).pow(2).mean()
        value_loss2 = (q2 - target_q).pow(2).mean()

        self.critic1_optim.zero_grad()
        value_loss1.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critic_network.parameters(), self.max_grad_norm)
        sync_grads(self.critic_network)
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        value_loss2.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critic_network2.parameters(), self.max_grad_norm)
        sync_grads(self.critic_network2)
        self.critic2_optim.step()

        # update actor

        with torch.no_grad():
            q1 = self.critic_target_network(inputs_norm_tensor, actions_tensor)
            q2 = self.critic_target_network2(inputs_norm_tensor, actions_tensor)
            q = torch.min(q1, q2)
            v = self.v_critic(inputs_norm_tensor)
            exp_a = torch.exp((q - v) * self.iql_temperature)
            exp_a = torch.clip(exp_a, 0, 100.0)
        action_real = self.actor_network(inputs_norm_tensor)
        # actions, logpi = self.actor_network.sample(inputs_norm_tensor)
        actor_loss = torch.mean(exp_a *  torch.square(action_real - actions_tensor))

        self.actor_optim.zero_grad()
        actor_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.actor_network.parameters(), self.max_grad_norm)
        sync_grads(self.actor_network)
        self.actor_optim.step()

    def _soft_update(self):
        # self._soft_update_target_network(self.actor_target_network, self.actor_network)
        self._soft_update_target_network(self.critic_target_network, self.critic_network)
        self._soft_update_target_network(self.critic_target_network2, self.critic_network2)
