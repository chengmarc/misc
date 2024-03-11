# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 19:37:49 2023

@author: Admin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %%
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

df = pd.read_csv('time_series.csv')

# %%
# data preprocessing
# split data into small chunks since my computer can't handle the whole dataset
new_df = df.iloc[:100]
# remove the first column
new_df.drop(new_df.columns[0], axis=1, inplace=True)
# convert nan to 0
new_df.fillna(0, inplace=True)
# maybe try keep date column and use it as a feature

# %%
# setup some parameters
forcast_steps = 1
winodw_size = 10

# %%
def create_inout_sequences(df, winodw_size, forcast_steps):
    # create a new dataframe with the required columns
    X, y = [], []
    for i in tqdm(range(len(df) - winodw_size - forcast_steps)):
        # keep the time column and time order
        X.append(df.iloc[i:i+winodw_size].values)
        y.append(df.iloc[i+winodw_size:i+winodw_size+forcast_steps].values)
    X, y = np.array(X), np.array(y)
    return X, y

# %%
inputs, targets = create_inout_sequences(new_df, winodw_size, forcast_steps)

    # %%
train_size = int(0.8 * len(inputs))
train_data, train_labels = inputs[:train_size], targets[:train_size]
test_data, test_labels = inputs[train_size:], targets[train_size:]

# %%
# use torch dataloader to load data
train_data = torch.from_numpy(train_data).float()
train_labels = torch.from_numpy(train_labels).float()
train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_data = torch.from_numpy(test_data).float()
test_labels = torch.from_numpy(test_labels).float()
test_dataset = TensorDataset(test_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# %%
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq, forcast_steps):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out)
        return predictions[:, -forcast_steps:, :]
    
# %%
# setup hyperparameters
num_samples, num_features = train_data.shape[0], train_data.shape[2]
hidden_layer_size = 64
output_size = forcast_steps
num_epochs = 100

# %%
criterion = nn.MSELoss()
model = LSTM(num_features, hidden_layer_size, output_size)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %%
for epoch in tqdm(range(num_epochs)):
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(data, forcast_steps)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch} Loss {loss.item()}')
    
# %%
model.eval()
with torch.no_grad():
    for batch_idx, (data, labels) in enumerate(test_loader):
        data = data.to(device)
        labels = labels.to(device)
        outputs = model(data, forcast_steps)
        loss = criterion(outputs, labels)
        print(f'Loss {loss.item()}')
