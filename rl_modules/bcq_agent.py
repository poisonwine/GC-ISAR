import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.base_agent import BaseAgent
from rl_modules.replay_buffer import replay_buffer
from rl_modules.models import actor, critic, SacActor
from mpi_utils.normalizer import normalizer
from her_modules.her import her_sampler
import torch.nn as nn
import torch.nn.functional as F
from rl_modules.models import VAE
from utils.net_utils import MLP, Perturbation

"""
ddpg with HER (MPI-version)

"""
class BCQ(BaseAgent):
    def __init__(self, args, env, env_params):
        super().__init__(args, env, env_params)
  
        # create the network
        
        self.actor_network = None
    
        self.critic_network = critic(env_params)
        self.critic_target_network = critic(env_params)

        self.critic_network2 = critic(env_params)
        self.critic_target_network2 = critic(env_params)

        # sync the networks across the cpus
     
        sync_networks(self.critic_network)
        sync_networks(self.critic_network2)
        # load the weights into the target networks
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        self.critic_target_network2.load_state_dict(self.critic_network2.state_dict())
        # if use gpu
        if self.args.cuda:
            self.critic_network.cuda()
            self.critic_target_network.cuda()
            self.critic_network2.cuda()
            self.critic_target_network2.cuda()
        # create the optimizer
        self.actor_optim = None
        self.critic1_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        self.critic2_optim = torch.optim.Adam(self.critic_network2.parameters(), lr=self.args.lr_critic)
        self.max_a = env_params['action_max']
        self.max_grad_norm = self.args.max_grad_norm
        self.num_repeat_actions = self.args.repeat_action
        self.lmbda = self.args.lmbda
        self.vae = None
        self.vae_optim = None
        self.bcq_phi = self.args.bcq_phi
        self.construct_vae_and_actor()


    def construct_vae_and_actor(self):
        state_dim = self.env_params['obs'] + self.env_params['goal']
        action_dim = self.env_params['action']
        
        if self.args.cuda:
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        net_a = MLP(input_dim=state_dim + action_dim,
                    output_dim=action_dim,
                    hidden_sizes=[256, 256],
                    device=device)
        self.actor_network = Perturbation(net_a, max_action=self.max_a, device=device, phi=self.bcq_phi).to(device)
        sync_networks(self.actor_network)
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)


        vae_encoder = MLP(input_dim= state_dim + action_dim,
                          hidden_sizes=self.args.vae_hidden_sizes,
                          device=device)
        
        vae_decoder = MLP(
            input_dim= state_dim + self.args.latent_dim,
            output_dim=action_dim,
            hidden_sizes=self.args.vae_hidden_sizes,
            device=device,
        )
        self.vae = VAE(
            vae_encoder,
            vae_decoder,
            hidden_dim=self.args.vae_hidden_sizes[-1],
            latent_dim=self.args.latent_dim,
            max_action=self.max_a,
            device=device,
        ).to(device)
        self.vae_optim = torch.optim.Adam(self.vae.parameters(), lr=self.args.vae_lr)
        

    def _stochastic_actions(self, input_tensor):
        decode_action = self.vae.decode(input_tensor)
        perturbed_act = self.actor_network(input_tensor, decode_action)
        action = perturbed_act.cpu().numpy().squeeze()
        return action
    
    def _deterministic_action(self, input_tensor):
        action  = self.vae.decode(input_tensor)
        return action

    # update the network
    def _update_network(self, future_p=None):
        # sample the episodes
        sample_batch = self.sample_batch(future_p=future_p)
        transitions = sample_batch['transitions']
        # start to do the update
        inputs_norm_tensor, inputs_next_norm_tensor, actions_tensor, r_tensor = self.process_transitions(transitions)
        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()


        # update vae
        recon, mean, std = self.vae(inputs_norm_tensor, actions_tensor)
        recon_loss = F.mse_loss(actions_tensor, recon)
        # (....) is D_KL( N(mu, sigma) || N(0,1) )
        KL_loss = (-torch.log(std) + (std.pow(2) + mean.pow(2) - 1) / 2).mean()
        vae_loss = recon_loss + KL_loss / 2
        self.vae_optim.zero_grad()
        vae_loss.backward()
        self.vae_optim.step()

        with torch.no_grad():
            # repeat num_sampled_action times
            obs_next = inputs_next_norm_tensor.repeat_interleave(self.num_repeat_actions, dim=0)
            # now obs_next: (num_sampled_action * batch_size, state_dim)

            # perturbed action generated by VAE
            act_next = self.vae.decode(obs_next)
            # now obs_next: (num_sampled_action * batch_size, action_dim)
            target_Q1 = self.critic_target_network(obs_next, act_next)
            target_Q2 = self.critic_target_network2(obs_next, act_next)

            # Clipped Double Q-learning
            target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1 - self.lmbda) * torch.max(target_Q1, target_Q2)
            # now target_Q: (num_sampled_action * batch_size, 1)

            # the max value of Q
            batch_size = inputs_norm_tensor.shape[0]
            target_Q = target_Q.reshape(batch_size, -1).max(dim=1)[0].reshape(-1, 1)
            # now target_Q: (batch_size, 1)

            target_Q = r_tensor  +  self.args.gamma * target_Q.detach()
            clip_return = 1 / (1 - self.args.gamma)
            target_Q = torch.clamp(target_Q, -clip_return, 0)

        

        current_Q1 = self.critic_network(inputs_norm_tensor, actions_tensor)
        current_Q2 = self.critic_network2(inputs_norm_tensor, actions_tensor)

        critic1_loss = F.mse_loss(current_Q1, target_Q)
        critic2_loss = F.mse_loss(current_Q2, target_Q)

        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critic_network.parameters(), self.max_grad_norm)
        sync_grads(self.critic_network)
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critic_network2.parameters(), self.max_grad_norm)
        sync_grads(self.critic_network2)
        self.critic2_optim.step()


        sampled_act = self.vae.decode(inputs_norm_tensor)
        perturbed_act = self.actor_network(inputs_norm_tensor, sampled_act)

        # max
        actor_loss = -self.critic_network(inputs_norm_tensor, perturbed_act).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.actor_network.parameters(), self.max_grad_norm)
        sync_grads(self.actor_network)
        self.actor_optim.step()


    def _soft_update(self):
        self._soft_update_target_network(self.critic_target_network, self.critic_network)
        self._soft_update_target_network(self.critic_target_network2, self.critic_network2)
