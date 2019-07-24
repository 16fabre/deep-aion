import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Reshape(nn.Module):
    def __init__(self, new_shape):
        super(Reshape, self).__init__()
        self.new_shape = new_shape

    def forward(self, x):
        x = x.view(-1, self.new_shape[0], self.new_shape[1])
        return x

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



#defining some params
config = {
    'data' : 'data/ECG5000/unsupervised/',
    'timesteps' : 140,
    'input_dim' : 1,
    'optimizer' : 'adam',
    'loss' : 'mse',
    'epochs' : 100,
    'batch_size' : 64,
    'gpu' : False,
    'standardize' : True,
    'normalize' : False,
    'units_enc' : 128,
    'units_dec' : 128,
    'latent_dim' : 2,
    'num_classes' : 1    
}

model = MLP_AE(config).to(device)
distance = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-5)

#load data
train = np.load(config['data'] + 'train.npy')
test = np.load(config['data'] + 'test.npy')
train_label = np.load(config['data'] + 'train_label.npy')
test_label = np.load(config['data'] + 'test_label.npy')

for epoch in range(config['epochs']):
    for i in range(train.shape[0] // config['batch_size']):
        x_train = train[i*config['batch_size']:(i+1)*config['batch_size']]
        x_train = torch.from_numpy(x_train).float().to(device)
        x_train = Variable(x_train)
        # ===================forward=====================
        output = model(x_train)
        loss = distance(output, x_train)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, config['epochs'], loss.item()))

#evaluate the model
test_pred = model(torch.from_numpy(test).float().to(device))
test_pred = test_pred.detach().cpu().numpy()
# np.save('test_pred.npy', test_pred)

plt.plot(test[0])
plt.plot(test_pred[0])
plt.legend(['Truth', 'Reconstruction'])
plt.savefig('img/reconstruction.png', bbox_inches='tight')