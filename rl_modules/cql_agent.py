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
"""
ddpg with HER (MPI-version)

"""
class CQL(BaseAgent):
    def __init__(self, args, env, env_params):
        super().__init__(args, env, env_params)
  

        # create the network
        self.actor_network = SacActor(env_params)
        # self.actor_target_network = SacActor(env_params)

        self.critic_network = critic(env_params)
        self.critic_target_network = critic(env_params)

        self.critic_network2 = critic(env_params)
        self.critic_target_network2 = critic(env_params)

        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)
        sync_networks(self.critic_network2)
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
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic1_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        self.critic2_optim = torch.optim.Adam(self.critic_network2.parameters(), lr=self.args.lr_critic)
        self.max_a = env_params['action_max']
        self.max_grad_norm = self.args.max_grad_norm
        self.num_repeat_actions = self.args.repeat_action
        self.cql_weight = 1.0
        self.cql_temp = 1.0
        self.cql_clip_diff_min = -np.inf
        self.cql_clip_diff_max = np.inf
        self.cql_log_alpha = torch.tensor([0.0], requires_grad=True)
        self.lagrange_threshold = self.args.lagrange_threshold
        self.cql_alpha_optimizer = torch.optim.Adam([self.cql_log_alpha], lr=1e-4)
        self.cql_lagrange = self.args.cql_lagrange
        self.cql_importance_sample = self.args.cql_importance_sample

        self.alpha_min = 0.0
        self.alpha_max = 1e6
        self.is_auto_alpha = self.args.is_auto_alpha
        if not self.is_auto_alpha:
            self.alpha = self.args.sac_alpha
        else:
            self.log_alpha = torch.tensor(0.0, requires_grad=True)
            self.alpha = self.log_alpha.detach().exp()
            self.target_entropy = -torch.prod(torch.FloatTensor(self.env.action_space.shape)).item()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-4, eps=1e-4)

    def _stochastic_actions(self, input_tensor):
        pi, log_sigma = self.actor_network(input_tensor)
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
        action, _ = self.actor_network(input_tensor)
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

        # update actor 
        actions_real, log_pi = self.actor_network.sample(inputs_norm_tensor)
        sample_q1 = self.critic_network(inputs_norm_tensor, actions_real)
        sample_q2 = self.critic_network2(inputs_norm_tensor, actions_real)
        policy_loss = -(torch.min(sample_q1, sample_q2) - self.alpha * log_pi)
        policy_loss = policy_loss.mean()

        self.actor_optim.zero_grad()
        policy_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.actor_network.parameters(), self.max_grad_norm)
        self.actor_optim.step()
        if self.is_auto_alpha:
            entropy_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            entropy_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.detach().exp()


        with torch.no_grad():
            action_next, new_log_pi = self.actor_network.sample(inputs_next_norm_tensor)
          
            target_q_value1 = self.critic_target_network(inputs_next_norm_tensor, action_next)
            target_q_value2 = self.critic_target_network2(inputs_next_norm_tensor, action_next)

            target_q = torch.min(target_q_value1, target_q_value2) - self.alpha * new_log_pi
        
            # q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
            target_q_value = r_tensor + self.args.gamma * target_q.detach()
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)


        # update critic 
        real_q_value1 = self.critic_network(inputs_norm_tensor, actions_tensor)
        critic_loss1 = (real_q_value1 - target_q_value).pow(2).mean()
        real_q_value2 = self.critic_network2(inputs_norm_tensor, actions_tensor)
        critic_loss2 = (real_q_value2 - target_q_value).pow(2).mean()

        batch_size = actions_tensor.shape[0]
        action_dim = actions_tensor.shape[-1]
        cql_random_actions = actions_tensor.new_empty(
            (batch_size, self.num_repeat_actions, action_dim), requires_grad=False).uniform_(-1, 1)

        obs = self.extend_and_repeat(inputs_norm_tensor, 1, self.num_repeat_actions)
        cql_current_actions, cql_current_log_pis = self.actor_network.sample(obs)
        # cql_current_action: [batchsize, self.repeat, action_dim] 
        # log_pis: [batch_size, repeat_num]
        next_obs = self.extend_and_repeat(inputs_next_norm_tensor, 1, self.num_repeat_actions)
        cql_next_actions, cql_next_log_pis = self.actor_network.sample(next_obs)
       
        cql_current_actions, cql_current_log_pis = (cql_current_actions.detach(),cql_current_log_pis.detach())
        cql_next_actions, cql_next_log_pis = (cql_next_actions.detach(),cql_next_log_pis.detach())

        # print(cql_next_log_pis.shape)
        cql_q1_rand = self.critic_network(obs, cql_random_actions).reshape(batch_size, -1) # (batch_size, self.num_repeats)
        cql_q2_rand = self.critic_network2(obs, cql_random_actions).reshape(batch_size, -1)
        cql_q1_current_actions = self.critic_network(obs, cql_current_actions).reshape(batch_size, -1)
        cql_q2_current_actions = self.critic_network2(obs, cql_current_actions).reshape(batch_size, -1)
        cql_q1_next_actions = self.critic_network(obs, cql_next_actions).reshape(batch_size, -1)
        cql_q2_next_actions = self.critic_network2(obs, cql_next_actions).reshape(batch_size, -1)

        # print(cql_q1_rand.shape, real_q_value1.shape, cql_q1_next_actions.shape, cql_q1_current_actions.shape)
        cql_cat_q1 = torch.cat([cql_q1_rand, real_q_value1, cql_q1_next_actions, cql_q1_current_actions,], dim=1)
        # print(cql_q1_rand.shape, real_q_value1.shape, cql_q1_next_actions.shape, cql_q1_current_actions.shape)
        cql_cat_q2 = torch.cat([cql_q2_rand, real_q_value2,cql_q2_next_actions,cql_q2_current_actions], dim=1)

        cql_std_q1 = torch.std(cql_cat_q1, dim=1)
        cql_std_q2 = torch.std(cql_cat_q2, dim=1)


        if self.cql_importance_sample:
            random_density = np.log(0.5**action_dim)
            cql_cat_q1 = torch.cat(
                [
                    cql_q1_rand - random_density,
                    cql_q1_next_actions - cql_next_log_pis.detach(),
                    cql_q1_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )

            cql_cat_q2 = torch.cat(
                [
                    cql_q2_rand - random_density,
                    cql_q2_next_actions - cql_next_log_pis.detach(),
                    cql_q2_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )
        cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.cql_temp, dim=1) * self.cql_temp
        cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.cql_temp, dim=1) * self.cql_temp


        cql_qf1_diff = torch.clamp(
            cql_qf1_ood - real_q_value1,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()
        cql_qf2_diff = torch.clamp(
            cql_qf2_ood - real_q_value2,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()


        if self.cql_lagrange:
            alpha_prime = torch.clamp(torch.exp(self.log_alpha), min=0.0, max=1e6)
            cql_qf1_diff = alpha_prime * (cql_qf1_diff - self.lagrange_threshold)  # noqa
            
            cql_qf2_diff = alpha_prime * (cql_qf2_diff - self.lagrange_threshold)  # noqa
              

            self.cql_alpha_optimizer.zero_grad()
            alpha_prime_loss = -(cql_qf1_diff + cql_qf2_diff) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.cql_alpha_optimizer.step()
        
        critic_loss1 = critic_loss1 + cql_qf1_diff
        critic_loss2 = critic_loss2 + cql_qf2_diff

        self.critic1_optim.zero_grad()
        critic_loss1.backward(retain_graph=True)
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critic_network.parameters(), self.max_grad_norm)
        sync_grads(self.critic_network)
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic_loss2.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critic_network2.parameters(), self.max_grad_norm)
        sync_grads(self.critic_network2)
        self.critic2_optim.step()
        


    def extend_and_repeat(self, tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
        return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)

    def _soft_update(self):
        self._soft_update_target_network(self.critic_target_network, self.critic_network)
        self._soft_update_target_network(self.critic_target_network2, self.critic_network2)
