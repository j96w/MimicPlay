"""
Config for MimicPlay algorithm.
"""

from mimicplay.configs.base_config import BaseConfig


class MimicPlayConfig(BaseConfig):
    ALGO_NAME = "mimicplay"

    def train_config(self):
        """
        MimicPlay doesn't need "next_obs" from hdf5 - so save on storage and compute by disabling it.
        """
        super(MimicPlayConfig, self).train_config()
        self.train.hdf5_load_next_obs = False

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the 
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config` 
        argument to the constructor. Any parameter that an algorithm needs to determine its 
        training and test-time behavior should be populated here.
        """

        # optimization parameters
        self.algo.optim_params.policy.optimizer_type = "adam"
        self.algo.optim_params.policy.learning_rate.initial = 1e-4      # policy learning rate
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.policy.learning_rate.epoch_schedule = [] # epochs where LR decay occurs
        self.algo.optim_params.policy.learning_rate.scheduler_type = "multistep" # learning rate scheduler ("multistep", "linear", etc) 
        self.algo.optim_params.policy.regularization.L2 = 0.00          # L2 regularization strength

        # loss weights
        self.algo.loss.l2_weight = 1.0      # L2 loss weight
        self.algo.loss.l1_weight = 0.0      # L1 loss weight
        self.algo.loss.cos_weight = 0.0     # cosine loss weight

        # MLP network architecture (layers after observation encoder and RNN, if present)
        self.algo.actor_layer_dims = (1024, 1024)

        # stochastic Gaussian policy settings
        self.algo.gaussian.enabled = False              # whether to train a Gaussian policy
        self.algo.gaussian.fixed_std = False            # whether to train std output or keep it constant
        self.algo.gaussian.init_std = 0.1               # initial standard deviation (or constant)
        self.algo.gaussian.min_std = 0.01               # minimum std output from network
        self.algo.gaussian.std_activation = "softplus"  # activation to use for std output from policy net
        self.algo.gaussian.low_noise_eval = True        # low-std at test-time 

        # stochastic GMM policy settings
        self.algo.gmm.enabled = False                   # whether to train a GMM policy
        self.algo.gmm.num_modes = 5                     # number of GMM modes
        self.algo.gmm.min_std = 0.0001                  # minimum std output from network
        self.algo.gmm.std_activation = "softplus"       # activation to use for std output from policy net
        self.algo.gmm.low_noise_eval = True             # low-std at test-time 

        # stochastic VAE policy settings
        self.algo.vae.enabled = False                   # whether to train a VAE policy
        self.algo.vae.latent_dim = 14                   # VAE latent dimnsion - set to twice the dimensionality of action space
        self.algo.vae.latent_clip = None                # clip latent space when decoding (set to None to disable)
        self.algo.vae.kl_weight = 1.                    # beta-VAE weight to scale KL loss relative to reconstruction loss in ELBO

        # VAE decoder settings
        self.algo.vae.decoder.is_conditioned = True                         # whether decoder should condition on observation
        self.algo.vae.decoder.reconstruction_sum_across_elements = False    # sum instead of mean for reconstruction loss

        # VAE prior settings
        self.algo.vae.prior.learn = False                                   # learn Gaussian / GMM prior instead of N(0, 1)
        self.algo.vae.prior.is_conditioned = False                          # whether to condition prior on observations
        self.algo.vae.prior.use_gmm = False                                 # whether to use GMM prior
        self.algo.vae.prior.gmm_num_modes = 10                              # number of GMM modes
        self.algo.vae.prior.gmm_learn_weights = False                       # whether to learn GMM weights 
        self.algo.vae.prior.use_categorical = False                         # whether to use categorical prior
        self.algo.vae.prior.categorical_dim = 10                            # the number of categorical classes for each latent dimension
        self.algo.vae.prior.categorical_gumbel_softmax_hard = False         # use hard selection in forward pass
        self.algo.vae.prior.categorical_init_temp = 1.0                     # initial gumbel-softmax temp
        self.algo.vae.prior.categorical_temp_anneal_step = 0.001            # linear temp annealing rate
        self.algo.vae.prior.categorical_min_temp = 0.3                      # lowest gumbel-softmax temp

        self.algo.vae.encoder_layer_dims = (300, 400)                       # encoder MLP layer dimensions
        self.algo.vae.decoder_layer_dims = (300, 400)                       # decoder MLP layer dimensions
        self.algo.vae.prior_layer_dims = (300, 400)                         # prior MLP layer dimensions (if learning conditioned prior)

        # RNN policy settings
        self.algo.rnn.enabled = False                               # whether to train RNN policy
        self.algo.rnn.horizon = 10                                  # unroll length for RNN - should usually match train.seq_length
        self.algo.rnn.hidden_dim = 400                              # hidden dimension size
        self.algo.rnn.rnn_type = "LSTM"                             # rnn type - one of "LSTM" or "GRU"
        self.algo.rnn.num_layers = 2                                # number of RNN layers that are stacked
        self.algo.rnn.open_loop = False                             # if True, action predictions are only based on a single observation (not sequence)
        self.algo.rnn.kwargs.bidirectional = False                  # rnn kwargs
        self.algo.rnn.kwargs.do_not_lock_keys()

        # GMM highlevel polict settings
        self.algo.highlevel.enabled = False                     # whether to train the highlevel planner of MimicPlay
        self.algo.highlevel.ac_dim = 30                         # 3D trajectory output dimension (3 x 10 points = 30)
        self.algo.highlevel.latent_plan_dim = 400               # latent plan embedding size
        self.algo.highlevel.do_not_lock_keys()

        # GPT lowlevel policy settings
        self.algo.lowlevel.enabled = False                      # whether to train the lowlevel guided policy of MimicPlay (if highlevel is not enabled, an end-to-end lowlevel policy will be trained (BC-transformer baseline))
        self.algo.lowlevel.feat_dim = 656                       # feature dimansion of transformer
        self.algo.lowlevel.n_layer = 4                          # number of layers in transformer
        self.algo.lowlevel.n_head = 4                           # number of heads in transformer
        self.algo.lowlevel.block_size = 10                      # sequence block size, which should be same as train.seq_length in json config file
        self.algo.lowlevel.gmm_modes = 5                        # number of gmm modes for action output
        self.algo.lowlevel.action_dim = 7                       # robot action dimension
        self.algo.lowlevel.proprio_dim = 7                      # input robot's proprioception dimension (end-effector 3D location + end-effector orientation in quaternion)
        self.algo.lowlevel.spatial_softmax_num_kp = 64          # number of keypoints in the spatial softmax pooling layer
        self.algo.lowlevel.gmm_min_std = 0.0001                 # std of gmm output during inference
        self.algo.lowlevel.dropout = 0.1                        # training dropout rate
        self.algo.lowlevel.trained_highlevel_planner = None     # load trained highlevel latent planner (set to None when learning highlevel planner or other baselines)
        self.algo.lowlevel.eval_goal_img_window = 30            # goal image sampling window during evaluation rollouts
        self.algo.lowlevel.eval_max_goal_img_iter = 5           # maximum idling steps of sampled goal image during evaluation rollouts
        self.algo.lowlevel.do_not_lock_keys()

        # Playdata training/inference settings
        self.algo.playdata.enable = False                       # whether to train with plan data (unlabeled, no-cut)
        self.algo.playdata.goal_image_range = [100, 200]        # goal image sampling range during training
        self.algo.playdata.eval_goal_gap = 150                  # goal image sampling gap during evaluation rollouts (mid of training goal_image_range)
        self.algo.playdata.do_not_lock_keys()

