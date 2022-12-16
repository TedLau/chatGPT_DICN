import ipaddress

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import pandas as pd
from dgl.nn.pytorch import SAGEConv


def load_data(filename):
    # Read the data from the text file
    df = pd.read_csv(filename, sep='	', header=None, names=['source ip', 'destination ip', 'weight', 'date'])

    # Preprocess the data
    graphs, labels = preprocess_data(df)

    return graphs, labels

def preprocess_data(df):
    # Group the data by date
    grouped = df.groupby('date')

    # Extract the features and labels for each group
    graphs = []
    labels = []
    for date, group in grouped:
        # Extract the IP addresses and weights

        group['source ip'] = group['source ip'].apply(lambda x: int(ipaddress.IPv4Address(x)))
        group['destination ip'] = group['destination ip'].apply(lambda x: int(ipaddress.IPv4Address(x)))
        src = group['source ip'].values
        dst = group['destination ip'].values
        weight = group['weight'].values
        # Convert the string weights to floats
        # weights = pd.to_numeric(weight, errors='coerce')
        #
        # # Calculate the mean weight
        # mean_weight = weights.mean()
        # Replace missing weights with the mean weight
        mean_weight = 0.13471230717108976
        weight = [mean_weight if w == 'None' else w for w in weight]
        lst = [float(x) for x in weight]

        # Create a tensor from the list
        weight = torch.tensor(lst).float()
        # Convert the weights to a float tensor
        # weight = torch.tensor(weight).float()
        unique_nodes = set(src) | set(dst)
        # label = len(unique_nodes)
        # Create a graph for this group
        g = dgl.DGLGraph()
        g.add_nodes(len(unique_nodes))
        g.add_edges(src, dst)
        g.edata['weight'] = weight

        # Extract the label for this group
        # unique_nodes = set(src) | set(dst)
        label = len(unique_nodes)

        graphs.append(g)
        labels.append(label)

    return graphs, labels

# Define the GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList([
            SAGEConv(in_feats, hidden_size, 'mean'),
            SAGEConv(hidden_size, hidden_size, 'mean'),
            SAGEConv(hidden_size, out_feats, 'mean')
        ])

    def forward(self, g, inputs):
        h = inputs
        for layer in self.layers:
            h = layer(g, h)
        return h

# Define the training and evaluation functions
def train(model, g, inputs, labels):
    # Set the model to training mode
    model.train()
    # Use the model to make predictions
    logits = model(g, inputs)
    # Calculate the loss
    loss = F.cross_entropy(logits, labels)
    # Clear the gradients
    optimizer.zero_grad()
    # Backpropagate the loss
    loss.backward()
    # Update the model parameters
    optimizer.step()

    return loss

def evaluate(model, g, inputs, labels):
    # Set the model to evaluation mode
    model.eval()
    # Use the model to make predictions
    logits = model(g, inputs)
    # Calculate the MAPE
    mape = ((logits - labels).abs() / labels).mean()
    return mape


# Load the data
graphs, labels = load_data('/Users/tedlau/PostGraduatePeriod/Graduate/Tasks/Task9-神经网络比较-不会--但又给续上了/data_by_day_simple/datatest.txt')

# Create the model
model = GraphSAGE(in_feats=1, hidden_size=16, out_feats=2)

# Use Adam as the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Set the number of epochs
num_epochs = 10

# Set the device to use for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model and data to the device
model = model.to(device)
# graphs = graphs.to(device)
# labels = labels.to(device)

# Loop over the epochs
for epoch in range(num_epochs):
    # Loop over the graphs and labels
    for g, label in zip(graphs, labels):
        # Extract the node features
        inputs = g.edata['weight']
        # Reshape the node features to (batch size, feature size)
        inputs = inputs.view(-1, 1)
        # Train the model on this graph
        loss = train(model, g, inputs, label)
        # Calculate the accuracy on the training set
        accuracy = evaluate(model, graphs, inputs, labels)
        print(f'Epoch {epoch+1}: loss={loss:.4f}, accuracy={accuracy:.4f}')

