"""
Contains torch Modules for policy networks. These networks take an
observation dictionary as input (and possibly additional conditioning,
such as subgoal or goal dictionaries) and produce action predictions,
samples, or distributions as outputs. Note that actions
are assumed to lie in [-1, 1], and most networks will have a final
tanh activation to help ensure this range.
"""
import textwrap
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import robomimic.utils.tensor_utils as TensorUtils
from robomimic.models.base_nets import Module
from robomimic.models.transformers import GPT_Backbone
from robomimic.models.vae_nets import VAE
from robomimic.models.distributions import TanhWrappedDistribution
from robomimic.models.policy_nets import RNNActorNetwork
from robomimic.models.obs_nets import RNN_MIMO_MLP

from mimicplay.models.obs_nets import MIMO_MLP

class ActorNetwork(MIMO_MLP):
    """
    A basic policy network that predicts actions from observations.
    Can optionally be goal conditioned on future observations.
    """
    def __init__(
        self,
        obs_shapes,
        ac_dim,
        mlp_layer_dims,
        goal_shapes=None,
        encoder_kwargs=None,
    ):
        """
        Args:
            obs_shapes (OrderedDict): a dictionary that maps observation keys to
                expected shapes for observations.

            ac_dim (int): dimension of action space.

            mlp_layer_dims ([int]): sequence of integers for the MLP hidden layers sizes.

            goal_shapes (OrderedDict): a dictionary that maps observation keys to
                expected shapes for goal observations.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-observation key information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """
        assert isinstance(obs_shapes, OrderedDict)
        self.obs_shapes = obs_shapes
        self.ac_dim = ac_dim

        # set up different observation groups for @MIMO_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)

        self._is_goal_conditioned = False
        if goal_shapes is not None and len(goal_shapes) > 0:
            assert isinstance(goal_shapes, OrderedDict)
            self._is_goal_conditioned = True
            self.goal_shapes = OrderedDict(goal_shapes)
            observation_group_shapes["goal"] = OrderedDict(self.goal_shapes)
        else:
            self.goal_shapes = OrderedDict()

        output_shapes = self._get_output_shapes()
        super(ActorNetwork, self).__init__(
            input_obs_group_shapes=observation_group_shapes,
            output_shapes=output_shapes,
            layer_dims=mlp_layer_dims,
            encoder_kwargs=encoder_kwargs,
        )

    def _get_output_shapes(self):
        """
        Allow subclasses to re-define outputs from @MIMO_MLP, since we won't
        always directly predict actions, but may instead predict the parameters
        of a action distribution.
        """
        return OrderedDict(action=(self.ac_dim,))

    def output_shape(self, input_shape=None):
        return [self.ac_dim]

    def forward(self, obs_dict, goal_dict=None):
        actions = super(ActorNetwork, self).forward(obs=obs_dict, goal=goal_dict)["action"]
        # apply tanh squashing to ensure actions are in [-1, 1]
        return torch.tanh(actions)

    def _to_string(self):
        """Info to pretty print."""
        return "action_dim={}".format(self.ac_dim)

class GMMActorNetwork(ActorNetwork):
    """
    Variant of actor network that learns a multimodal Gaussian mixture distribution
    over actions.
    """
    def __init__(
        self,
        obs_shapes,
        ac_dim,
        mlp_layer_dims,
        num_modes=5,
        min_std=0.01,
        std_activation="softplus",
        low_noise_eval=True,
        use_tanh=False,
        goal_shapes=None,
        encoder_kwargs=None,
    ):
        """
        Args:
            obs_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for observations.

            ac_dim (int): dimension of action space.

            mlp_layer_dims ([int]): sequence of integers for the MLP hidden layers sizes.

            num_modes (int): number of GMM modes

            min_std (float): minimum std output from network

            std_activation (None or str): type of activation to use for std deviation. Options are:

                `'softplus'`: Softplus activation applied

                `'exp'`: Exp applied; this corresponds to network output being interpreted as log_std instead of std

            low_noise_eval (float): if True, model will sample from GMM with low std, so that
                one of the GMM modes will be sampled (approximately)

            use_tanh (bool): if True, use a tanh-Gaussian distribution

            goal_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for goal observations.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """

        # parameters specific to GMM actor
        self.num_modes = num_modes
        self.min_std = min_std
        self.low_noise_eval = low_noise_eval
        self.use_tanh = use_tanh

        # Define activations to use
        self.activations = {
            "softplus": F.softplus,
            "exp": torch.exp,
        }
        assert std_activation in self.activations, \
            "std_activation must be one of: {}; instead got: {}".format(self.activations.keys(), std_activation)
        self.std_activation = std_activation

        super(GMMActorNetwork, self).__init__(
            obs_shapes=obs_shapes,
            ac_dim=ac_dim,
            mlp_layer_dims=mlp_layer_dims,
            goal_shapes=goal_shapes,
            encoder_kwargs=encoder_kwargs,
        )

    def _get_output_shapes(self):
        """
        Tells @MIMO_MLP superclass about the output dictionary that should be generated
        at the last layer. Network outputs parameters of GMM distribution.
        """
        return OrderedDict(
            mean=(self.num_modes, self.ac_dim), 
            scale=(self.num_modes, self.ac_dim), 
            logits=(self.num_modes,),
        )

    def forward_train(self, obs_dict, goal_dict=None, return_latent=False):
        """
        Return full GMM distribution, which is useful for computing
        quantities necessary at train-time, like log-likelihood, KL 
        divergence, etc.

        Args:
            obs_dict (dict): batch of observations
            goal_dict (dict): if not None, batch of goal observations

        Returns:
            dist (Distribution): GMM distribution
        """
        if return_latent:
            out, enc_out, mlp_out = MIMO_MLP.forward(self, return_latent=return_latent, obs=obs_dict, goal=goal_dict)
        else:
            out = MIMO_MLP.forward(self, return_latent=return_latent, obs=obs_dict, goal=goal_dict)
        means = out["mean"]
        scales = out["scale"]
        logits = out["logits"]

        # apply tanh squashing to means if not using tanh-GMM to ensure means are in [-1, 1]
        if not self.use_tanh:
            means = torch.tanh(means)

        # Calculate scale
        if self.low_noise_eval and (not self.training):
            # low-noise for all Gaussian dists
            scales = torch.ones_like(means) * 1e-4
        else:
            # post-process the scale accordingly
            scales = self.activations[self.std_activation](scales) + self.min_std

        # mixture components - make sure that `batch_shape` for the distribution is equal
        # to (batch_size, num_modes) since MixtureSameFamily expects this shape
        component_distribution = D.Normal(loc=means, scale=scales)
        component_distribution = D.Independent(component_distribution, 1)

        # unnormalized logits to categorical distribution for mixing the modes
        mixture_distribution = D.Categorical(logits=logits)

        dist = D.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )

        if self.use_tanh:
            # Wrap distribution with Tanh
            dist = TanhWrappedDistribution(base_dist=dist, scale=1.)

        if return_latent:
            return dist, enc_out, mlp_out
        return dist

    def forward(self, obs_dict, goal_dict=None):
        """
        Samples actions from the policy distribution.

        Args:
            obs_dict (dict): batch of observations
            goal_dict (dict): if not None, batch of goal observations

        Returns:
            action (torch.Tensor): batch of actions from policy distribution
        """
        dist = self.forward_train(obs_dict, goal_dict)
        return dist.sample()

    def _to_string(self):
        """Info to pretty print."""
        return "action_dim={}\nnum_modes={}\nmin_std={}\nstd_activation={}\nlow_noise_eval={}".format(
            self.ac_dim, self.num_modes, self.min_std, self.std_activation, self.low_noise_eval)

class RNNGMMActorNetwork(RNNActorNetwork):
    """
    An RNN GMM policy network that predicts sequences of action distributions from observation sequences.
    """

    def __init__(
            self,
            obs_shapes,
            ac_dim,
            mlp_layer_dims,
            rnn_hidden_dim,
            rnn_num_layers,
            rnn_type="LSTM",  # [LSTM, GRU]
            rnn_kwargs=None,
            num_modes=5,
            min_std=0.01,
            std_activation="softplus",
            low_noise_eval=True,
            use_tanh=False,
            goal_shapes=None,
            encoder_kwargs=None,
    ):
        """
        Args:

            rnn_hidden_dim (int): RNN hidden dimension

            rnn_num_layers (int): number of RNN layers

            rnn_type (str): [LSTM, GRU]

            rnn_kwargs (dict): kwargs for the torch.nn.LSTM / GRU

            num_modes (int): number of GMM modes

            min_std (float): minimum std output from network

            std_activation (None or str): type of activation to use for std deviation. Options are:

                `'softplus'`: Softplus activation applied

                `'exp'`: Exp applied; this corresponds to network output being interpreted as log_std instead of std

            low_noise_eval (float): if True, model will sample from GMM with low std, so that
                one of the GMM modes will be sampled (approximately)

            use_tanh (bool): if True, use a tanh-Gaussian distribution

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """

        # parameters specific to GMM actor
        self.num_modes = num_modes
        self.min_std = min_std
        self.low_noise_eval = low_noise_eval
        self.use_tanh = use_tanh

        # Define activations to use
        self.activations = {
            "softplus": F.softplus,
            "exp": torch.exp,
        }
        assert std_activation in self.activations, \
            "std_activation must be one of: {}; instead got: {}".format(self.activations.keys(), std_activation)
        self.std_activation = std_activation

        super(RNNGMMActorNetwork, self).__init__(
            obs_shapes=obs_shapes,
            ac_dim=ac_dim,
            mlp_layer_dims=mlp_layer_dims,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
            rnn_type=rnn_type,
            rnn_kwargs=rnn_kwargs,
            goal_shapes=goal_shapes,
            encoder_kwargs=encoder_kwargs,
        )

    def _get_output_shapes(self):
        """
        Tells @MIMO_MLP superclass about the output dictionary that should be generated
        at the last layer. Network outputs parameters of GMM distribution.
        """
        return OrderedDict(
            mean=(self.num_modes, self.ac_dim),
            scale=(self.num_modes, self.ac_dim),
            logits=(self.num_modes,),
        )

    def forward_train(self, obs_dict, goal_dict=None, rnn_init_state=None, return_state=False):
        """
        Return full GMM distribution, which is useful for computing
        quantities necessary at train-time, like log-likelihood, KL
        divergence, etc.

        Args:
            obs_dict (dict): batch of observations
            goal_dict (dict): if not None, batch of goal observations
            rnn_init_state: rnn hidden state, initialize to zero state if set to None
            return_state (bool): whether to return hidden state

        Returns:
            dists (Distribution): sequence of GMM distributions over the timesteps
            rnn_state: return rnn state at the end if return_state is set to True
        """

        outputs = RNN_MIMO_MLP.forward(
            self, obs=obs_dict, goal=goal_dict, rnn_init_state=rnn_init_state, return_state=return_state)

        if return_state:
            outputs, state = outputs
        else:
            state = None

        means = outputs["mean"]
        scales = outputs["scale"]
        logits = outputs["logits"]

        # apply tanh squashing to mean if not using tanh-GMM to ensure means are in [-1, 1]
        if not self.use_tanh:
            means = torch.tanh(means)

        if self.low_noise_eval and (not self.training):
            # low-noise for all Gaussian dists
            scales = torch.ones_like(means) * 1e-4
        else:
            # post-process the scale accordingly
            scales = self.activations[self.std_activation](scales) + self.min_std

        # mixture components - make sure that `batch_shape` for the distribution is equal
        # to (batch_size, timesteps, num_modes) since MixtureSameFamily expects this shape
        component_distribution = D.Normal(loc=means, scale=scales)
        component_distribution = D.Independent(component_distribution, 1)  # shift action dim to event shape

        # unnormalized logits to categorical distribution for mixing the modes
        mixture_distribution = D.Categorical(logits=logits)

        dists = D.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )

        if self.use_tanh:
            # Wrap distribution with Tanh
            dists = TanhWrappedDistribution(base_dist=dists, scale=1.)

        if return_state:
            return dists, state
        else:
            return dists

    def forward(self, obs_dict, goal_dict=None, rnn_init_state=None, return_state=False):
        """
        Samples actions from the policy distribution.

        Args:
            obs_dict (dict): batch of observations
            goal_dict (dict): if not None, batch of goal observations

        Returns:
            action (torch.Tensor): batch of actions from policy distribution
        """
        out = self.forward_train(obs_dict=obs_dict, goal_dict=goal_dict, rnn_init_state=rnn_init_state,
                                 return_state=return_state)
        if return_state:
            ad, state = out
            return ad.sample(), state
        return out.sample()

    def forward_train_step(self, obs_dict, goal_dict=None, rnn_state=None):
        """
        Unroll RNN over single timestep to get action GMM distribution, which
        is useful for computing quantities necessary at train-time, like
        log-likelihood, KL divergence, etc.

        Args:
            obs_dict (dict): batch of observations. Should not contain
                time dimension.
            goal_dict (dict): if not None, batch of goal observations
            rnn_state: rnn hidden state, initialize to zero state if set to None

        Returns:
            ad (Distribution): GMM action distributions
            state: updated rnn state
        """
        obs_dict = TensorUtils.to_sequence(obs_dict)
        ad, state = self.forward_train(
            obs_dict, goal_dict, rnn_init_state=rnn_state, return_state=True)

        # to squeeze time dimension, make another action distribution
        assert ad.component_distribution.base_dist.loc.shape[1] == 1
        assert ad.component_distribution.base_dist.scale.shape[1] == 1
        assert ad.mixture_distribution.logits.shape[1] == 1
        component_distribution = D.Normal(
            loc=ad.component_distribution.base_dist.loc.squeeze(1),
            scale=ad.component_distribution.base_dist.scale.squeeze(1),
        )
        component_distribution = D.Independent(component_distribution, 1)
        mixture_distribution = D.Categorical(logits=ad.mixture_distribution.logits.squeeze(1))
        ad = D.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )
        return ad, state

    def forward_step(self, obs_dict, goal_dict=None, rnn_state=None):
        """
        Unroll RNN over single timestep to get sampled actions.

        Args:
            obs_dict (dict): batch of observations. Should not contain
                time dimension.
            goal_dict (dict): if not None, batch of goal observations
            rnn_state: rnn hidden state, initialize to zero state if set to None

        Returns:
            acts (torch.Tensor): batch of actions - does not contain time dimension
            state: updated rnn state
        """
        obs_dict = TensorUtils.to_sequence(obs_dict)
        acts, state = self.forward(
            obs_dict, goal_dict, rnn_init_state=rnn_state, return_state=True)
        assert acts.shape[1] == 1
        return acts[:, 0], state

    def _to_string(self):
        """Info to pretty print."""
        msg = "action_dim={}, std_activation={}, low_noise_eval={}, num_nodes={}, min_std={}".format(
            self.ac_dim, self.std_activation, self.low_noise_eval, self.num_modes, self.min_std)
        return msg

