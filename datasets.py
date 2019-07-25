from torch.utils import data
import numpy as np
import pandas as pd

class ECG5000(data.Dataset):
    def __init__(self, data_set='test', data_type='unsupervised'):
        self.data_type = data_type
        # Load csv file
        self.data = pd.read_csv('data/ECG5000/' + data_type + '/ECG_data.csv')
        # Select only the dataset requested (train, validation or test)
        self.data = self.data[self.data['set'] == data_set]
        self.labels = self.data['label'].values
        self.data = self.data.drop(columns = ['label', 'set']).values

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        X = self.data[idx][:,np.newaxis]
        y = self.labels[idx]
        if self.data_type == 'unsupervised':
            return X, X
        else:
            return X, y