import torch
import ml_casadi.torch as mlcs
import os
from collision_predictor_mpc import COLPREDMPC_WEIGHT_DIR


## def all default weight filenames here for convenience
SIGMOID_FILENAME = os.path.join(COLPREDMPC_WEIGHT_DIR, 'sigmoid_fitting')
RESNET_FILENAME = os.path.join(COLPREDMPC_WEIGHT_DIR, 'resnet8')
VAE_FILENAME = os.path.join(COLPREDMPC_WEIGHT_DIR, 'vae')
COLPRED_FILENAME = os.path.join(COLPREDMPC_WEIGHT_DIR, 'depth_state_check')


class NNBase(mlcs.TorchMLCasadiModule):
    """Base NN wrapper with convenience functions regarding device and saving.loading weights."""
    def __init__(self, filename):
        """Init the Colpred object with a model architecture and the weight filename.
        The device is set to cuda if available.
        """
        super().__init__()
        ## get GPU if available; else get CPU
        self._dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)
        self.filename = filename
        self.eval()
        self.zero_grad()


    @property
    def device(self):
        return self._dev
    @device.setter
    def device(self, value):
        assert value in ['cpu', 'cuda']
        self._dev = value
        self.to(value)
        for module in self.children():
            module.device = value


    def load_weights(self, filename=None):
        """Loads the weights from a given filename, or from self.filename by default.
        If the loading fails because the computer is cpu only, force loading onto cpu.
        """
        if filename is None: filename = self.filename
        try:
            self.load_state_dict(torch.load(filename + '.pth'))
        except RuntimeError:
            print('loading model weights on cpu')
            self.load_state_dict(torch.load(filename + '.pth', map_location='cpu'))


    def save_weights(self, filename=None):
        """Save the weights to a given filename, or to self.filename by default."""
        if filename is None: filename = self.filename
        torch.save(self.state_dict(), filename + '.pth')


    def save_weights_idx(self, idx):
        """Convenience function for saving weights in different files at each epoch.
        A folder named after self.filename is created, then files are named epoch_N.pth.
        """
        if not os.path.isdir(self.filename): os.makedirs(self.filename)
        outfile = self.filename + '/epoch_' + str(idx)
        self.save_weights(filename=outfile)


    def init_conv_layer(self, layer):
        """Recursive function to initialize all conv2d layers with xavier_uniform_ througout the submodules."""
        if type(layer) == torch.nn.Conv2d:
            torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain('conv2d'))
            torch.nn.init.zeros_(layer.bias)
        for ll in layer.children():
            self.init_conv_layer(ll)
