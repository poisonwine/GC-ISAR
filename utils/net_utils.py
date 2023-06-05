import numpy as np
import torch
from torch import nn
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    no_type_check,
)
ModuleType = Type[nn.Module]

ArgsType = Union[Tuple[Any, ...], Dict[Any, Any], Sequence[Tuple[Any, ...]],
                 Sequence[Dict[Any, Any]]]



def miniblock(
    input_size: int,
    output_size: int = 0,
    norm_layer: Optional[ModuleType] = None,
    norm_args: Optional[Union[Tuple[Any, ...], Dict[Any, Any]]] = None,
    activation: Optional[ModuleType] = None,
    act_args: Optional[Union[Tuple[Any, ...], Dict[Any, Any]]] = None,
    linear_layer: Type[nn.Linear] = nn.Linear,
) -> List[nn.Module]:
    """Construct a miniblock with given input/output-size, norm layer and \
    activation."""
    layers: List[nn.Module] = [linear_layer(input_size, output_size)]
    if norm_layer is not None:
        if isinstance(norm_args, tuple):
            layers += [norm_layer(output_size, *norm_args)]  # type: ignore
        elif isinstance(norm_args, dict):
            layers += [norm_layer(output_size, **norm_args)]  # type: ignore
        else:
            layers += [norm_layer(output_size)]  # type: ignore
    if activation is not None:
        if isinstance(act_args, tuple):
            layers += [activation(*act_args)]
        elif isinstance(act_args, dict):
            layers += [activation(**act_args)]
        else:
            layers += [activation()]
    return layers


class MLP(nn.Module):
    """Simple MLP backbone.

    Create a MLP of size input_dim * hidden_sizes[0] * hidden_sizes[1] * ...
    * hidden_sizes[-1] * output_dim

    :param int input_dim: dimension of the input vector.
    :param int output_dim: dimension of the output vector. If set to 0, there
        is no final linear layer.
    :param hidden_sizes: shape of MLP passed in as a list, not including
        input_dim and output_dim.
    :param norm_layer: use which normalization before activation, e.g.,
        ``nn.LayerNorm`` and ``nn.BatchNorm1d``. Default to no normalization.
        You can also pass a list of normalization modules with the same length
        of hidden_sizes, to use different normalization module in different
        layers. Default to no normalization.
    :param activation: which activation to use after each layer, can be both
        the same activation for all layers if passed in nn.Module, or different
        activation for different Modules if passed in a list. Default to
        nn.ReLU.
    :param device: which device to create this model on. Default to None.
    :param linear_layer: use this module as linear layer. Default to nn.Linear.
    :param bool flatten_input: whether to flatten input data. Default to True.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 0,
        hidden_sizes: Sequence[int] = (),
        norm_layer: Optional[Union[ModuleType, Sequence[ModuleType]]] = None,
        norm_args: Optional[ArgsType] = None,
        activation: Optional[Union[ModuleType, Sequence[ModuleType]]] = nn.ReLU,
        act_args: Optional[ArgsType] = None,
        device: Optional[Union[str, int, torch.device]] = None,
        linear_layer: Type[nn.Linear] = nn.Linear,
        flatten_input: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        if norm_layer:
            if isinstance(norm_layer, list):
                assert len(norm_layer) == len(hidden_sizes)
                norm_layer_list = norm_layer
                if isinstance(norm_args, list):
                    assert len(norm_args) == len(hidden_sizes)
                    norm_args_list = norm_args
                else:
                    norm_args_list = [norm_args for _ in range(len(hidden_sizes))]
            else:
                norm_layer_list = [norm_layer for _ in range(len(hidden_sizes))]
                norm_args_list = [norm_args for _ in range(len(hidden_sizes))]
        else:
            norm_layer_list = [None] * len(hidden_sizes)
            norm_args_list = [None] * len(hidden_sizes)
        if activation:
            if isinstance(activation, list):
                assert len(activation) == len(hidden_sizes)
                activation_list = activation
                if isinstance(act_args, list):
                    assert len(act_args) == len(hidden_sizes)
                    act_args_list = act_args
                else:
                    act_args_list = [act_args for _ in range(len(hidden_sizes))]
            else:
                activation_list = [activation for _ in range(len(hidden_sizes))]
                act_args_list = [act_args for _ in range(len(hidden_sizes))]
        else:
            activation_list = [None] * len(hidden_sizes)
            act_args_list = [None] * len(hidden_sizes)
        hidden_sizes = [input_dim] + list(hidden_sizes)
        model = []
        for in_dim, out_dim, norm, norm_args, activ, act_args in zip(
            hidden_sizes[:-1], hidden_sizes[1:], norm_layer_list, norm_args_list,
            activation_list, act_args_list
        ):
            model += miniblock(
                in_dim, out_dim, norm, norm_args, activ, act_args, linear_layer
            )
        if output_dim > 0:
            model += [linear_layer(hidden_sizes[-1], output_dim)]
        self.output_dim = output_dim or hidden_sizes[-1]
        self.model = nn.Sequential(*model)
        self.flatten_input = flatten_input

    @no_type_check
    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if self.device is not None:
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if self.flatten_input:
            obs = obs.flatten(1)
        return self.model(obs)

class Perturbation(nn.Module):
    """Implementation of perturbation network in BCQ algorithm. Given a state and \
    action, it can generate perturbed action.

    :param torch.nn.Module preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param float max_action: the maximum value of each dimension of action.
    :param Union[str, int, torch.device] device: which device to create this model on.
        Default to cpu.
    :param float phi: max perturbation parameter for BCQ. Default to 0.05.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        You can refer to `examples/offline/offline_bcq.py` to see how to use it.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        max_action: float,
        device: Union[str, int, torch.device] = "cpu",
        phi: float = 0.05
    ):
        # preprocess_net: input_dim=state_dim+action_dim, output_dim=action_dim
        super(Perturbation, self).__init__()
        self.preprocess_net = preprocess_net
        self.device = device
        self.max_action = max_action
        self.phi = phi

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # preprocess_net
        logits = self.preprocess_net(torch.cat([state, action], -1))[0]
        noise = self.phi * self.max_action * torch.tanh(logits)
        # clip to [-max_action, max_action]
        return (noise + action).clamp(-self.max_action, self.max_action)


if __name__ == '__main__':
    vae_encoder = MLP(input_dim= 64,
                     hidden_sizes=[128,128],
                     device=torch.device('cpu'))
    print(vae_encoder)