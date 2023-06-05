import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union

"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.

"""

# Inverse tanh torch function
def atanh(z):
    return 0.5 * (torch.log(1 + z) - torch.log(1 - z))

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

# define the actor network
class actor(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])
        
        self.apply(weights_init_)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions

# define the actor network
class actor_inverse(nn.Module):
    def __init__(self, env_params):
        super(actor_inverse, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(3*env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])
        
        self.apply(weights_init_)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions

# define the planner network 
class planner(nn.Module):
    def __init__(self, env_params):
        super(planner, self).__init__()
        self.fc1 = nn.Linear(2 * env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['goal'])
        
        self.apply(weights_init_)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.action_out(x)

        return actions

class critic(nn.Module):
    def __init__(self, env_params, activation=None):
        super(critic, self).__init__()
        self.activation = activation
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

        self.apply(weights_init_)

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        if self.activation == 'sigmoid':
            q_value = torch.sigmoid(q_value)
        return q_value

class doublecritic(nn.Module):
    def __init__(self, env_params):
        super(doublecritic, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q1_out = nn.Linear(256, 1)

        self.fc4 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 256)
        self.q2_out = nn.Linear(256, 1)

        self.apply(weights_init_)

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=-1)
        x1 = F.relu(self.fc1(x))
        x1 = F.relu(self.fc2(x1))
        x1 = F.relu(self.fc3(x1))
        q1_value = self.q1_out(x1)

        x2 = F.relu(self.fc4(x))
        x2 = F.relu(self.fc5(x2))
        x2 = F.relu(self.fc6(x2))
        q2_value = self.q2_out(x2)

        return q1_value, q2_value
    
    def Q1(self, x, action):
        x = torch.cat([x, action / self.max_action], dim=-1)
        x1 = F.relu(self.fc1(x))
        x1 = F.relu(self.fc2(x1))
        x1 = F.relu(self.fc3(x1))
        q1_value = self.q1_out(x1)
        
        return q1_value

class value(nn.Module):
    def __init__(self, env_params):
        super(value, self).__init__()
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

        self.apply(weights_init_)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value

class SacActor(nn.Module):
    def __init__(self, env_params, min_log_sigma=-20.0, max_log_sigma=2.0):
        super(SacActor, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        # self.fc3 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, env_params['action'])
        self.fc_sigma = nn.Linear(256, env_params['action'])
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma
        # self.action_out = nn.Linear(256, env_params['action'])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        mu = self.fc_mu(x)
        log_sigma = self.fc_sigma(x)
        log_sigma = torch.clamp(log_sigma, self.min_log_sigma, self.max_log_sigma)
        return mu, log_sigma

    def sample(self, state):
        mu, log_sigma = self.forward(state)
        sigma = torch.exp(log_sigma)
        # if torch.sum(torch.isnan(mu)).item() > 0:
        #     print('meet Nan value!!')
        dist = Normal(mu, sigma)
        # * reparameterization trick: recognize the difference of sample() and rsample()
        action = dist.rsample()
        tanh_action = torch.tanh(action)
        # * the log-probabilities of actions can be calculated in closed forms
        log_prob = dist.log_prob(action)
        log_prob = (log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-6)).sum(-1)
        return tanh_action, log_prob



class VAE(nn.Module):
    """Implementation of VAE. It models the distribution of action. Given a \
    state, it can generate actions similar to those in batch. It is used \
    in BCQ algorithm.

    :param torch.nn.Module encoder: the encoder in VAE. Its input_dim must be
        state_dim + action_dim, and output_dim must be hidden_dim.
    :param torch.nn.Module decoder: the decoder in VAE. Its input_dim must be
        state_dim + latent_dim, and output_dim must be action_dim.
    :param int hidden_dim: the size of the last linear-layer in encoder.
    :param int latent_dim: the size of latent layer.
    :param float max_action: the maximum value of each dimension of action.
    :param Union[str, torch.device] device: which device to create this model on.
        Default to "cpu".

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        You can refer to `examples/offline/offline_bcq.py` to see how to use it.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        hidden_dim: int,
        latent_dim: int,
        max_action: float,
        device: Union[str, torch.device] = "cpu"
    ):
        super(VAE, self).__init__()
        self.encoder = encoder

        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)

        self.decoder = decoder

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [state, action] -> z , [state, z] -> action
        latent_z = self.encoder(torch.cat([state, action], -1))
        # shape of z: (state.shape[:-1], hidden_dim)

        mean = self.mean(latent_z)
        # Clamped for numerical stability
        log_std = self.log_std(latent_z).clamp(-4, 15)
        std = torch.exp(log_std)
        # shape of mean, std: (state.shape[:-1], latent_dim)

        latent_z = mean + std * torch.randn_like(std)  # (state.shape[:-1], latent_dim)

        reconstruction = self.decode(state, latent_z)  # (state.shape[:-1], action_dim)
        return reconstruction, mean, std

    def decode(
        self,
        state: torch.Tensor,
        latent_z: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:
        # decode(state) -> action
        if latent_z is None:
            # state.shape[0] may be batch_size
            # latent vector clipped to [-0.5, 0.5]
            latent_z = torch.randn(state.shape[:-1] + (self.latent_dim, )) \
                .to(self.device).clamp(-0.5, 0.5)

        # decode z with state!
        return self.max_action * \
            torch.tanh(self.decoder(torch.cat([state, latent_z], -1)))

class BehaviorPriorNet(nn.Module):
    def __init__(self, env_params):
        super(BehaviorPriorNet, self).__init__()
        self.state_dim = env_params['obs']
        self.goal_dim = env_params['goal']
        self.action_dim = env_params['action']
        self.encoder = nn.Sequential(nn.Linear(self.state_dim+self.goal_dim, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, self.state_dim + self.goal_dim))

        self.predictor = nn.Sequential(nn.Linear(self.state_dim + self.goal_dim, 128),
                                       nn.ReLU(),
                                       nn.Linear(128, self.action_dim))
    
    def forward(self, x):
        x = self.encoder(x)
        actions = self.predictor(x)
        return actions


if __name__ == '__main__':
    env_params =  {'action_max':1.0,
                    'goal': 3, 'obs':10,'action':4}
    actor = SacActor(env_params=env_params)
    s = torch.rand(64,32,10)
    g = torch.rand(64,32,3)
    obs = torch.cat((s,g), dim=-1)
    action, log = actor.sample(obs)
    print(log.shape, action.shape)
