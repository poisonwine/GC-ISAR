import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.base_agent import BaseAgent
from rl_modules.replay_buffer import replay_buffer
from rl_modules.models import actor, critic, value, BehaviorPriorNet
from mpi_utils.normalizer import normalizer
from her_modules.her import her_sampler

import torch.nn as nn
import torch.nn.functional as F
"""
ddpg with HER (MPI-version)

"""
class BPR(BaseAgent):
    def __init__(self, args, env, env_params):
        super().__init__(args, env, env_params) 
  

        # create the network
        self.actor_network = actor(env_params)
        self.actor_target_network = actor(env_params)

        self.critic_network = critic(env_params)
        self.critic_target_network = critic(env_params)

        self.critic_network2 = critic(env_params)
        self.critic_target_network2 = critic(env_params)

        self.critic_v = value(env_params)

        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)
        sync_networks(self.critic_network2)
        sync_networks(self.critic_v)
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        self.critic_target_network2.load_state_dict(self.critic_network2.state_dict())
        # if use gpu
        if self.args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()
            self.critic_network2.cuda()
            self.critic_target_network2.cuda()
            self.critic_v.cuda()
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic1_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        self.critic2_optim = torch.optim.Adam(self.critic_network2.parameters(), lr=self.args.lr_critic)
        self.critic_v_optim = torch.optim.Adam(self.critic_v.parameters(), lr=self.args.lr_critic)
        self.noisy_range = self.args.noise_range
        self.policy_noise = self.args.policy_noise
        self.max_a = env_params['action_max']
        self.total_iter = 0
        self.alpha = self.args.alpha
        self.policy_delay = self.args.policy_decay
        self.max_grad_norm = self.args.max_grad_norm

        self.bevior_network = BehaviorPriorNet(env_params)
        self.bevior_network_optim = torch.optim.Adam(self.bevior_network.parameters(), lr=self.args.lr_actor)
        sync_networks(self.bevior_network)
        # self.target_Q = 
        self.expectile = self.args.expectile

    def _expectile_regression(self, diff: torch.Tensor) -> torch.Tensor:
        weight = torch.where(diff > 0, self.expectile, (1 - self.expectile))
        return weight * (diff**2)


    # this function will choose action for the agent and do the exploration
    def _stochastic_actions(self, input_tensor):
        with torch.no_grad():
            input_repre = self.bevior_network.encoder(input_tensor)
        pi = self.actor_network(input_repre)
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
        with torch.no_grad():
            input_repre = self.bevior_network.encoder(input_tensor)
        action = self.actor_network(input_repre)
        return action

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

        norm_action = self.action_norm.normalize(actions_tensor.cpu().numpy())
        norm_action = torch.tensor(norm_action, dtype=torch.float32)

        predict_action = self.bevior_network(inputs_norm_tensor)
        pred_norm_action = torch.clamp((predict_action - torch.tensor(self.action_norm.mean, dtype=torch.float32)) / torch.tensor(self.action_norm.std, dtype=torch.float32), -5.0, 5.0) 

        bevior_loss = F.mse_loss(norm_action, pred_norm_action)

        self.bevior_network_optim.zero_grad()
        bevior_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.bevior_network.parameters(), self.max_grad_norm)
        sync_grads(self.bevior_network)
        self.bevior_network_optim.step()

        # update value network
        encoder = self.bevior_network.encoder
        with torch.no_grad():
            # encoder = self.bevior_network.encoder
            q_min = torch.min(self.critic_target_network(encoder(inputs_norm_tensor), actions_tensor),
                              self.critic_target_network2(encoder(inputs_norm_tensor), actions_tensor))
            input_represent = encoder(inputs_norm_tensor).detach()
        
        current_v = self.critic_v(input_represent)
        critic_v_loss = self._expectile_regression(q_min - current_v).mean()

        self.critic_v_optim.zero_grad()
        critic_v_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critic_v.parameters(), self.max_grad_norm)
        sync_grads(self.critic_v)
        self.critic_v_optim.step()


        # calculate the target Q value function
        with torch.no_grad():
            
            # target_next_action = self.actor_target_network(inputs_next_norm_tensor)
            # noise = (torch.rand_like(actions_tensor) * self.policy_noise).clamp(-self.noisy_range, self.noisy_range)
            # target_next_action = target_next_action + noise
            # target_next_action = torch.clamp(target_next_action, -self.max_a, self.max_a)
            # q_min = torch.min(self.critic_target_network(inputs_next_norm_tensor, target_next_action),
            #                   self.critic_target_network2(inputs_next_norm_tensor, target_next_action))
            # target_q_value = r_tensor + self.args.gamma * q_min.detach()
            # clip_return = 1 / (1 - self.args.gamma)
            # target_q_value = torch.clamp(target_q_value, -clip_return, 0)
            input_next_represent = encoder(inputs_next_norm_tensor).detach()
            next_v = self.critic_v(input_next_represent)
            target_q_value = r_tensor + self.args.gamma * next_v.detach()
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
            



        # update critic 
        real_q_value1 = self.critic_network(input_represent, actions_tensor)
        value_loss1 = (real_q_value1 - target_q_value).pow(2).mean() 
        real_q_value2 = self.critic_network2(input_represent, actions_tensor)
        value_loss2 = (real_q_value2 - target_q_value).pow(2).mean()

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
        if self.total_iter % self.policy_delay == 0:

            current_action = self.actor_network(input_represent)
            q = self.critic_network(input_represent, current_action)

            with torch.no_grad():
                v = self.critic_v(input_represent)
                actions_next = self.actor_network(input_next_represent)
                # q = r_tensor + self.args.gamma * self.critic_target_network(inputs_next_norm_tensor, actions_next)
                q = r_tensor + self.args.gamma * torch.min(self.critic_target_network(input_next_represent, actions_next),
                                                           self.critic_target_network2(input_next_represent, actions_next))
                adv = q - v
                weights = torch.clamp(torch.exp(adv), 0, 10.0)

            lmbda = self.alpha / q.abs().mean().detach()
            actor_loss = -lmbda * q.mean() + torch.mean(weights * torch.square(current_action - actions_tensor))
            self.actor_optim.zero_grad()
            actor_loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.actor_network.parameters(), self.max_grad_norm)
            sync_grads(self.actor_network)
            self.actor_optim.step()
        self.total_iter += 1
           

    def _soft_update(self):
        self._soft_update_target_network(self.actor_target_network, self.actor_network)
        self._soft_update_target_network(self.critic_target_network, self.critic_network)
        self._soft_update_target_network(self.critic_target_network2, self.critic_network2)
