# Import necessary libraries

import numpy as np
import dgl
import torch
import dgl.nn as nn
# Define a function to preprocess the data
import numpy as np


def preprocess_data(data):
    """Preprocess the data for training and testing.

    Args:
        data: A 2D numpy array with the source nodes in the first column,
            the destination nodes in the second column, the edge weights in the
            third column, and the dates in the last column.

    Returns:
        A list with the training data and a list with the testing data.
    """
    # Convert the source and destination nodes to integers
    data[:, 0] = data[:, 0]#.astype(int)
    data[:, 1] = data[:, 1]#.astype(int)

    # Convert the edge weights to floats
    data[:, 2] = data[:, 2].astype(float)

    # Sort the data by date
    data = data[data[:, 3].argsort()]

    # Split the data into training and testing sets
    train_data, test_data = data[:int(len(data) * 0.8)], data[int(len(data) * 0.8):]

    return list(train_data), list(test_data)


def load_data():
    with open('/data_by_day_simple/datatest.txt', 'r') as f:
        return [line.strip().split() for line in f]
# Load the data from a file or database
data = np.array(load_data())

# Preprocess the data for training and testing
train_data, test_data = preprocess_data(data)
# Split the preprocessed data into training and testing sets
# The training set contains the data from the first 6 days
# The testing set contains the data from the 7th day
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

# Convert the training data and testing data to lists of tuples
train_data = [(src, dst, weight) for src, dst, weight, date in train_data]
test_data = [(src, dst, weight) for src, dst, weight, date in test_data]
# Create a dictionary to map IP addresses to integer IDs
ip_to_id = {}
id_to_ip = {}
next_id = 0
for src, dst, weight, date in data:
    if src not in ip_to_id:
        ip_to_id[src] = next_id
        id_to_ip[next_id] = src
        next_id += 1
    if dst not in ip_to_id:
        ip_to_id[dst] = next_id
        id_to_ip[next_id] = dst
        next_id += 1

# Convert the IP addresses to integer IDs
train_data = [(ip_to_id[src], ip_to_id[dst], weight) for src, dst, weight in train_data]
test_data = [(ip_to_id[src], ip_to_id[dst], weight) for src, dst, weight in test_data]

# Create the DGL graph from the training data
g = dgl.graph(train_data, num_nodes=len(np.unique(data[:, :2])))

# Set the labels for each node in the graph
g.ndata['label'] = torch.tensor(data[:, 3], dtype=torch.float)

# Define the GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(in_feats, hidden_size),
            nn.ReLU()
        ])
        self.classify = nn.Linear(hidden_size, num_classes)

    def forward(self, g, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return self.classify(inputs)

# Initialize the GraphSAGE model
model = GraphSAGE(in_feats=1, hidden_size=64, num_classes=1)

# Set the model to the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    # Forward pass
    predictions = model(g, g.ndata['label'])
    loss = loss_fn(predictions, g.ndata['label'])

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss
    print('Epoch {}: Loss = {}'.format(epoch, loss.item()))

# Test the model on the testing data
test_g = dgl.graph(test_data, num_nodes=len(np.unique(data[:, :2])))
test_g.ndata['label'] = torch.tensor(data[:, 3], dtype=torch.float)
test_predictions = model(test_g, test_g.ndata['label'])
test_loss = loss_fn(test_predictions, test_g.ndata['label'])
print('Test Loss = {}'.format(test_loss.item()))

#
