"""
https://github.com/exe1023/LSTM_LN/blob/master/lstm.py

Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton 
https://arxiv.org/abs/1607.06450
"""

import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# %%

# setup some parameters
forecast_steps = 10
window_size = 100

# setup path for database based on google drive
path = r"C:\Users\uzcheng\Desktop\database"

files = os.listdir(path)

files = [os.path.join(path, file) for file in files]
file = files[0]

# %%
def find_sequence(df, sequence_length):
    sequence = [[] for _ in range(sequence_length)]
    df.sort_index(inplace=True)
    for i in range(len(df) - sequence_length + 1):
        split_df = df[i: (i + sequence_length)]
        # check if there is any nan in the sequence
        if split_df.isnull().values.any():
            continue
        for j in range(sequence_length):
            # append the values to the corresponding sequence
            lst_val = split_df.iloc[j].values.tolist()
            sequence[j].append(lst_val[0])
    return sequence

df = pd.read_csv(file, usecols = ["snapped_at", "price"])
df = df.set_index("snapped_at")
seq = find_sequence(df, window_size + forecast_steps)

# %%


# concat all the dataframes by both rows and columns
columns_name = ["price_" + str(i) for i in range(window_size + forecast_steps)]

# concat all dataframe into one
dataframes = pd.DataFrame(seq).T
dataframes.columns = columns_name

dataframes.to_csv(os.path.join(path, "tensor.csv"))

# %%
# save the final dataframe

# data preprocessing
# split data into small chunks since my computer can't handle the whole dataset
new_df = dataframes.iloc[:]

def create_inout_sequences(df, window_size):
    # first window_size columns are the input, the last forcast_steps columns are the output
    X = df.iloc[:, :window_size].values
    y = df.iloc[:, window_size:].values
    return X, y

inputs, targets = create_inout_sequences(new_df, window_size)

train_size = int(0.8 * len(inputs))
train_data, train_labels = inputs[:train_size], targets[:train_size]
test_data, test_labels = inputs[train_size:], targets[train_size:]


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

# setup hyperparameters
num_samples, num_features = train_data.shape
hidden_layer_size = 128
output_size = forecast_steps
num_epochs = 500
learning_rate = 1
num_samples, num_features

criterion = nn.MSELoss()

script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

# %%
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out)
        return predictions

criterion = nn.MSELoss()
model = LSTM(num_features, hidden_layer_size, output_size)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# %%
for epoch in tqdm(range(num_epochs)):
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch} Loss {loss.item()}')

# %%
# test the model
model.eval()
with torch.no_grad():
    for batch_idx, (data, labels) in enumerate(test_loader):
        data = data.to(device)
        labels = labels.to(device)
        outputs = model(data)
        loss = criterion(outputs, labels)
        print(f'Loss {loss.item()}')