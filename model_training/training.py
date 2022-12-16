import torch
import pandas as pd
import numpy as np
from model import LogisticRegression

# define the experimet index
exp_index = 1

# load the data
data = pd.read_csv(r'../raw_collected_data/demo{}.csv'.format(exp_index), header=None).to_numpy()

# derive missing ratio of the time series
window_size = 15
timesteps = data.shape[1] - 1
inputs = []
for index in range(9):
    time_series = data[index][1:]
    for i in range(timesteps-window_size):
        series = time_series[i:i+window_size]
        zeros = np.where(series == 0)[0]
        num_missing = len(zeros)
        inputs.append(num_missing)
inputs = np.array(inputs, dtype=float)
print(inputs.shape)

# derive target data
label_data = pd.read_csv(r'../raw_collected_data/labels{}.csv'.format(exp_index), header=None).to_numpy()[:,1:]
tags, time_length = label_data.shape
outputs = []
for index in range(tags):
    for j in range(time_length):
        outputs.append(label_data[index, j])
outputs = np.array(outputs)

# change data to tensor
inputs = torch.from_numpy(inputs).unsqueeze(-1).float()
outputs = torch.from_numpy(outputs).unsqueeze(-1).float()
assert inputs.shape[0] == outputs.shape[0]

# split the data
num_samples = inputs.shape[0]
train_inputs = inputs[:int(0.7*num_samples)]
val_inputs = inputs[int(0.7*num_samples):int(0.8*num_samples)]
test_inputs = inputs[int(0.8*num_samples):int(num_samples)]
train_outputs = outputs[:int(0.7*num_samples)]
val_outputs = outputs[int(0.7*num_samples):int(0.8*num_samples)]
test_outputs = outputs[int(0.8*num_samples):int(num_samples)]


# define training hyper-parameter
epoch = 30
learning_rate = 0.001
best_validation_loss = torch.inf

# perform training
model = LogisticRegression(1,1).float()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for i in range(epoch):

    # switch the mode to training
    print('training epoch:', i)
    model.train()

    # forward
    train_prediction = model(train_inputs)

    # calculate loss
    loss = criterion(train_prediction, train_outputs)

    # calculate gradient
    loss.backward() 
    
    # update parameters
    optimizer.step() 

    # calculate validation loss
    # switch the mode to testing
    model.eval()
    val_prediction = model(val_inputs)
    val_loss = criterion(val_prediction, val_outputs)
    if val_loss < best_validation_loss:
        torch.save(model.state_dict(), r'./best_model.pth')
        best_validation_loss = val_loss

