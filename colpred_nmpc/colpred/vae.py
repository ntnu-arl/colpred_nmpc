import numpy as np
import torch
from collision_predictor_mpc.colpred import NNBase, VAE_FILENAME


class Encoder(NNBase):
    """Image encoder into a latent space.
    The output shape is 2*size_latent since the VAE makes use of mean+std.
    A sample of size_latent dim is generated from this distribution and then decoder is used to reconstruct this image from the generated sample.
    """
    def __init__(self, size_latent, dropout_rate=0.2, filename=None):
        filename = filename if filename is not None else VAE_FILENAME
        filename += '_encoder'
        super().__init__(filename)
        self.nb_chan = 1  # depth images
        self.size_latent = size_latent
        self.dropout_rate = dropout_rate

        self.layers = torch.nn.ModuleDict({
            'input': torch.nn.Sequential(
                torch.nn.Conv2d(self.nb_chan, 32, kernel_size=(5,5), stride=2, padding=(1,3)),
                torch.nn.MaxPool2d(kernel_size=(3,3), stride=2),
            ),
            'res_layer_1': torch.nn.Conv2d(32, 32, kernel_size=(1,1), stride=2, padding=0),
            'res_block_1': torch.nn.Sequential(
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_rate),
                torch.nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_rate),
                torch.nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1),
            ),
            'res_layer_2': torch.nn.Conv2d(32, 64, kernel_size=(1,1), stride=2, padding=0),
            'res_block_2': torch.nn.Sequential(
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_rate),
                torch.nn.Conv2d(32, 64, kernel_size=(3,3), stride=2, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_rate),
                torch.nn.Conv2d(64, 64, kernel_size=(3,3), stride=1, padding=1),
            ),
            'res_layer_3': torch.nn.Conv2d(64, 128, kernel_size=(1,1), stride=2, padding=0),
            'res_block_3': torch.nn.Sequential(
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_rate),
                torch.nn.Conv2d(64, 128, kernel_size=(3,3), stride=2, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.Conv2d(128, 128, kernel_size=(3,3), stride=1, padding=1),
            ),
            'output': torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_rate),
                torch.nn.AvgPool2d(kernel_size=(3,3), stride=2),
                torch.nn.Flatten(),
                # torch.nn.Linear(3584, 512),
            ),
            'mean': torch.nn.Linear(3584, size_latent),
            'logvar': torch.nn.Linear(3584, size_latent),
        })

        self.init_conv_layer(self.layers)
        self.eval()
        self.zero_grad()


    def forward(self, input):
        x = self.layers['input'](input)
        res = self.layers['res_layer_1'](x)
        x = self.layers['res_block_1'](x)
        x = x + res
        res = self.layers['res_layer_2'](x)
        x = self.layers['res_block_2'](x)
        x = x + res
        res = self.layers['res_layer_3'](x)
        x = self.layers['res_block_3'](x)
        x = x + res
        x = self.layers['output'](x)
        mean = self.layers['mean'](x)
        logvar = self.layers['logvar'](x)
        return mean, logvar


    def encode(self, input):
        x = self.layers['input'](input)
        res = self.layers['res_layer_1'](x)
        x = self.layers['res_block_1'](x)
        x = x + res
        res = self.layers['res_layer_2'](x)
        x = self.layers['res_block_2'](x)
        x = x + res
        res = self.layers['res_layer_3'](x)
        x = self.layers['res_block_3'](x)
        x = x + res
        x = self.layers['output'](x)
        mean = self.layers['mean'](x)
        return mean


class Decoder(NNBase):
    """Image decoder from a latent space."""
    def __init__(self, size_latent, filename=None):
        filename = filename if filename is not None else VAE_FILENAME
        filename += '_decoder'
        super().__init__(filename)
        self.nb_chan = 1  # depth images
        self.size_latent = size_latent

        self.layers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(size_latent, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 128*9*15),
            ),
            torch.nn.Sequential(
                torch.nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=(2,2), output_padding=(0,1), dilation=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(64, 32, kernel_size=6, stride=4, padding=(2,2), output_padding=(0,0), dilation=1),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(32, 16, kernel_size=6, stride=2, padding=(0,0), output_padding=(0,1)),
                torch.nn.BatchNorm2d(16),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(16, self.nb_chan, kernel_size=4, stride=2, padding=2),
                torch.nn.Sigmoid(),
            )
        ])

        self.init_conv_layer(self.layers)
        self.eval()
        self.zero_grad()


    def forward(self, input):
        x = self.layers[0](input)
        x = x.view(x.size(0), 128, 9, 15)
        x = self.layers[1](x)
        return x



class VAE(NNBase):
    """Variational AutoEncoder."""
    def __init__(self, size_latent, dropout_rate=0.2, filename=None):
        filename = filename if filename is not None else VAE_FILENAME
        super().__init__(filename)
        self.nb_chan = 1  # depth images
        self.size_latent = size_latent
        self.encoder = Encoder(self.size_latent, dropout_rate, filename)
        self.decoder = Decoder(self.size_latent, filename)
        self.eval()
        self.zero_grad()

    def forward(self, input):
        ## encode
        mean, logvar = self.encoder(input)

        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z_sampled = eps * std + mean
        else:
            z_sampled = mean

        ## decode
        output = self.decoder(z_sampled)

        return output, mean, logvar
