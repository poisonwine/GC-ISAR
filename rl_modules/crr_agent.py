import torch
import torch.nn.functional as F
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.base_agent import BaseAgent
from rl_modules.models import actor, critic
from rl_modules.discriminator import Discriminator

"""
CRR (MPI-version)

"""
class CRR(BaseAgent):
    def __init__(self, args, env, env_params):
        super().__init__(args, env, env_params) 
        # create the network
        self.actor_network = actor(env_params)
        self.critic_network = critic(env_params)
        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)
        # build up the target network
        self.actor_target_network = actor(env_params)
        self.critic_target_network = critic(env_params)
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())


        # if use gpu
        if self.args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()
            # self.discriminator.cuda()

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        
        self.repeat_action = 10

    # this function will choose action for the agent and do the exploration
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

        # Compute the actions
        actions_real = self.actor_network(inputs_norm_tensor)
        # calculate the target Q value function
        # if self.args.method == 'awac':
        with torch.no_grad():
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)

            # the q loss
        real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()

        # Compute the advantage weighting
        with torch.no_grad():
            ## CRR mean
            obs = inputs_norm_tensor.unsqueeze(1).repeat_interleave(self.repeat_action, dim=1) # (batch, repeat, dim)
            action = self.actor_network(obs)
            q_values = self.critic_network(obs, action).reshape(inputs_norm_tensor.shape[0], -1) # (batch, repeat)
            q_mean = q_values.mean(-1).reshape(-1, 1)
            q_real = real_q_value.detach()
            weights = torch.clamp(torch.exp(q_real - q_mean), 0, 10.0)
            
            
        actor_loss = torch.mean(weights * torch.square(actions_real - actions_tensor))

        # update the actor network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network)
        self.actor_optim.step()

        # update the critic_network
       
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_network)
        self.critic_optim.step()
