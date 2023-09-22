import numpy as np
import torch
from collision_predictor_mpc.colpred import NNBase, RESNET_FILENAME


class ResNet8(NNBase):
    """Pytorch implementation of ResNet8 for processing the depth image.
    The output of the conv layers is flattened and compressed to a latent representation through some FC layers.
    """
    def __init__(self, size_latent, dropout_rate=0.2, filename=None):
        filename = filename if filename is not None else RESNET_FILENAME
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
                torch.nn.Linear(3584, size_latent),
            )
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
        return x


    def encode(self, input):
        return self.forward(input)
