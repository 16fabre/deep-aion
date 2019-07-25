import torch.nn as nn
from layers import Flatten, Reshape

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