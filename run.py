import torch
import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
from datasets import ECG5000
from models import MLP_AE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

# Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

# Defining configuration of the model
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
    'units_enc' : 64,
    'units_dec' : 64,
    'latent_dim' : 2,
    'num_classes' : 1    
}

params_loader = {
    'batch_size' : config['batch_size'],
    'shuffle':True,
    'num_workers':4
}

# Load data
train_set = ECG5000(data_set='train')
train_generator = data.DataLoader(train_set, **params_loader)
validation_set = ECG5000(data_set='validation')
validation_generator = data.DataLoader(validation_set, **params_loader)
test_set = ECG5000(data_set='test')

model = MLP_AE(config).to(device)
distance = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-5)
print(model)

for epoch in range(config['epochs']):
    print('Epoch {}/{}'.format(epoch + 1, config['epochs']))
    # Fit
    train_loss = 0.0
    model.train(True)
    for X_train, y_train in train_generator:
        X_train, y_train = X_train.float().to(device), y_train.float().to(device)
        # Forward
        output = model(X_train)
        loss = distance(output, y_train)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()
    
    # Validation
    val_loss = 0.0
    model.train(False)
    with torch.set_grad_enabled(False):
        for X_val, y_val in validation_generator:
            X_val, y_val = X_val.float().to(device), y_val.float().to(device)
            output = model(X_val)
            loss = distance(output, y_val)
            val_loss += loss.item()
    # Log
    print('Train loss: %.4f ; Validation loss: %.4f'%(train_loss / len(train_set), val_loss / len(validation_set)))
    print('-' * 10)

# Model inference on test data
model.train(False)
with torch.set_grad_enabled(False):
    X_test, y_test = test_set[0]
    print(X_test)
    X_test_pred = model(torch.from_numpy(X_test[np.newaxis, :, :]).float().to(device))
    print(X_test_pred)

plt.plot(X_test)
plt.plot(X_test_pred.cpu().numpy()[0])
plt.legend(['Truth', 'Reconstruction'])
plt.title('ECG MLP_AE - latent_dim 2')
plt.savefig('img/reconstruction.png', bbox_inches='tight')