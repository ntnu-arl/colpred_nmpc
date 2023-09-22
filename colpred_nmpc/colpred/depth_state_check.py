import numpy as np
import casadi as cs
import torch
import ml_casadi.torch.modules as mlcs
from collision_predictor_mpc.colpred import NNBase, COLPRED_FILENAME
from collision_predictor_mpc.colpred.resnet8 import ResNet8
from collision_predictor_mpc.colpred.vae import Encoder


class DepthStateChecker(NNBase):
    """MLP-like network written with ml-casadi for inference in NMPC.
    Computes a collision label for a given state and latent.
    Also returns a reconstruction of the state input to enforce the propagation of state info througout the network (vs the much larger latent vector).
    """
    def __init__(self, nb_states, size_latent, filename=None):
        filename = filename if filename is not None else COLPRED_FILENAME
        filename += '_linear'
        super().__init__(filename)
        self.nb_states = nb_states
        self.size_latent = size_latent

        self.layers = torch.nn.ModuleDict({
            'state_in': torch.nn.Sequential(
                mlcs.nn.Linear(nb_states, 16),
                mlcs.nn.activation.Tanh(),
            ),
            'main': torch.nn.Sequential(
                mlcs.nn.Linear(16+self.size_latent, 64),
                mlcs.nn.activation.Tanh(),
                mlcs.nn.Linear(64, 64),
                mlcs.nn.activation.Tanh(),
                mlcs.nn.Linear(64, 32),
                mlcs.nn.activation.Tanh(),
            ),
            'colpred': torch.nn.Sequential(
                mlcs.nn.Linear(32, 1),
            ),
        })

        self.eval()
        self.zero_grad()


    def forward(self, state, latent):
        x = self.layers['state_in'](state)
        x = torch.concatenate((x, latent), 1)
        x = self.layers['main'](x)
        x = self.layers['colpred'](x)
        x = torch.nn.functional.sigmoid(x)
        return x



class Colpred(NNBase):
    """NN-based collision predictor.
    The NN predicts the probability of a given state being in collision, given a depth image taken in another state configuration.
    The overall network is made of two parts:
    1- An encoder part which processes the depth image and compresses it to a latent space.
    2- A linear part made of FC layers which takes as input the robot state and combines it with the latent vector mid-way.
    Both parts are trained together and weights are stored individually.
    At inference time, the encoder NN processes images as they come, while the linear NN is integrated into the NMPC with the latent vector passed as parameters.
    """
    def __init__(self, nb_states=3, size_latent=128, dropout_rate=0.2, filename=None):
        filename = filename if filename is not None else COLPRED_FILENAME
        super().__init__(filename)
        self.size_latent = size_latent
        # self.encoder = ResNet8(size_latent, dropout_rate)
        self.encoder = Encoder(size_latent, dropout_rate, filename)
        self.linear = DepthStateChecker(nb_states, size_latent, filename)
        self.eval()
        self.zero_grad()


    def forward(self, state, depth):
        # latent = self.encoder.encode(depth)
        # return self.linear(state, latent)

        mean, logvar = self.encoder(depth)

        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z_sampled = eps * std + mean
        else:
            z_sampled = mean
        pred = self.linear(state, z_sampled)

        return pred, mean, logvar


    def infer_sigm(self, args):
        """NMPC inference function to fit the template in model.py.
        args[0] -- flag to disable inference,
        args[1] -- state to process,
        args[2] -- latent representation of the depth image.
        """
        x = self.linear.layers['state_in'](args[1])
        x = cs.vertcat(x, args[2])
        x = self.linear.layers['main'](x)
        x = self.linear.layers['colpred'](x)
        x = 1/(1+cs.exp(-x))
        return args[0]*x


    def infer_exp(self, args):
        """NMPC inference function to fit the template in model.py.
        args[0] -- flag to disable inference,
        args[1] -- state to process,
        args[2] -- latent representation of the depth image.
        """
        x = self.linear.layers['state_in'](args[1])
        x = cs.vertcat(x, args[2])
        x = self.linear.layers['main'](x)
        x = self.linear.layers['colpred'](x)
        x = cs.exp(x)/2
        return args[0]*x
