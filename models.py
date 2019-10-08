import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
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

class EncoderLSTM(nn.Module):
    def __init__(self, config):
        super(EncoderLSTM, self).__init__()
        self.__dict__.update(config)

        self.lstm = nn.LSTM(self.input_dim, self.units_enc, batch_first=True)
        
        #initialize weights with glorot uniform (like in Keras)
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=np.sqrt(6 / (self.input_dim + self.units_enc)))
        nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=np.sqrt(6 / (self.input_dim + self.units_dec)))
        
    def forward(self, x):
        x, hidden = self.lstm(x)
        hidden = Permute(1, 0, 2)(hidden[0])
        hidden = hidden.repeat(1, self.timesteps, 1)
        return hidden

class DecoderLSTM(nn.Module):
    def __init__(self, config):
        super(DecoderLSTM, self).__init__()
        self.__dict__.update(config)
        
        self.lstm = nn.LSTM(self.units_enc, self.units_dec, batch_first=True)
        self.dense = nn.Linear(self.units_dec, self.input_dim)

        #initialize weights
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=np.sqrt(6 / (self.units_enc + self.units_dec)))
        nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=np.sqrt(6 / (self.units_enc + self.units_dec)))
        
    def forward(self, x):
        x, hidden = self.lstm(x)
        x = self.dense(x)
        return x

class LSTM_AE(nn.Module):
    def __init__(self, config):
        super(LSTM_AE, self).__init__()
        self.__dict__.update(config)
        self.encoder = EncoderLSTM(config)
        self.decoder = DecoderLSTM(config)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class MLP_VAE(nn.Module):
    def __init__(self, config):
        super(MLP_VAE, self).__init__()
        self.__dict__.update(config)
        self.flat = Flatten()
        self.dense1 = nn.Linear(self.timesteps * self.input_dim, 64)
        self.densemu = nn.Linear(64, self.latent_dim)
        self.denselogvar = nn.Linear(64, self.latent_dim)
        self.dense2 = nn.Linear(self.latent_dim, 64)
        self.dense3 = nn.Linear(64, self.timesteps * self.input_dim)
        self.reshape = Reshape((self.timesteps, self.input_dim))
    
    def encoder(self, x):
        x = self.flat(x)
        x = self.dense1(x)
        mu, logvar = self.densemu(x), self.denselogvar(x)
        return mu, logvar

    def decoder(self, x):
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.reshape(x)
        return x
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x = self.decoder(z)
        return x, mu, logvar

def vae_loss(x_true, x_pred, mu, logvar):
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    
    loss = torch.nn.functional.mse_loss(x_true, x_pred, reduction='sum') # Binary Crossentropy
    loss += -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # Kullback–Leibler Divergence

    return loss