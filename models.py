import torch.nn as nn
from layers import Flatten, Reshape, Permute

class MLP_AE(nn.Module):
    def __init__(self, config):
        super(MLP_AE,self).__init__()
        self.__dict__.update(config)
        self.encoder = nn.Sequential(
            Flatten(),
            nn.Linear(self.timesteps * self.input_dim, self.units_enc),
            nn.Linear(self.units_enc, self.latent_dim),
            )
        self.decoder = nn.Sequential(             
            nn.Linear(self.latent_dim,self.units_dec),
            nn.Linear(self.units_dec, self.timesteps * self.input_dim),
            Reshape((self.timesteps, self.input_dim))
            )
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class CNN_AE(nn.Module):
    def __init__(self, config):
        super(CNN_AE, self).__init__()
        self.__dict__.update(config)
        self.encoder = nn.Sequential(
            Permute(0, 2, 1),
            nn.Conv1d(1, 16, 5, stride=2, padding=1), # (16, 70)
            nn.ReLU(True),
            nn.MaxPool1d(2), # (16, 35)
            nn.Conv1d(16, 8, 5, stride=2, padding=1), # (8, 17)
            nn.ReLU(True),
            nn.MaxPool1d(2), # (8, 8)
            nn.Conv1d(8, 2, 3, stride=1, padding=1), # (2, 8)
            nn.ReLU(True),
            nn.MaxPool1d(2) # (2, 4)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(2, 16, 5, stride=4), # (16, 17)
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 8, 5, stride=4), # (8, 69)
            nn.ReLU(True),
            nn.ConvTranspose1d(8, 1, 3, stride=2, output_padding=1), # (1, 140)
            Permute(0, 2, 1)
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x