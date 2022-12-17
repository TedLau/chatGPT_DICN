import dgl
import dgl.function as fn
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import torch
import dgl
import numpy as np


# Preprocessing function to convert the raw data into a graph and labels
def preprocess_data(df):
    df['src'] = df['src'].astype(str)
    df['dst'] = df['dst'].astype(str)
    df['date'] = df['date'].astype(str)
    unique_nodes = set(df['src'].unique()).union(df['dst'].unique())
    node_dict = {node: i for i, node in enumerate(unique_nodes)}
    df['src'] = df['src'].map(node_dict)
    df['dst'] = df['dst'].map(node_dict)
    df['weight'] = df['weight'].replace('None', np.nan)
    df['weight'] = df['weight'].astype(float)
    mean_weight = df['weight'].mean()
    df['weight'] = df['weight'].fillna(mean_weight)
    return df



# Load function to split the data into windows and create graphs and labels for each window
def load_data(filename, window_size):
    # Read the data into a Pandas dataframe and sort by date
    df = pd.read_csv(filename, sep='	', header=None, names=['src', 'dst', 'weight', 'date'])
    # preprocess data
    df = preprocess_data(df)

    # group data by date
    df_grouped = df.groupby('date')

    # create list of graphs for each date
    graphs = []
    labels = []
    for date, df_day in df_grouped:
        # create a graph for each day


        # add nodes and edges to the graph
        src = df_day['src'].values
        dst = df_day['dst'].values
        weight = df_day['weight'].values
        g = dgl.graph((src,dst),num_nodes=len(np.unique(np.concatenate((src, dst), axis=0))))
        # g.add_edges(src, dst)
        # g.add_nodes(len(np.unique(np.concatenate((src, dst), axis=0))))
        # g.add_nodes(len(np.unique(np.concatenate((src, dst), axis=0))))
        # set node and edge features
        g.ndata['weight'] = weight

        # append graph to list of graphs
        graphs.append(g)

        # get unique node count for the day and add it to the labels list
        unique_nodes = len(np.unique(np.concatenate((src, dst), axis=0)))
        labels.append(unique_nodes)

    # create a sliding window of graphs and labels
    graphs = [graphs[i:i + window_size] for i in range(len(graphs) - window_size)]
    labels = labels[window_size:]

    return graphs, labels

# Training function
def train(model, optimizer, graphs, labels):
    # Set model to training mode
    model.train()
    # Loop through the graphs and labels and optimize the model
    for g, l in zip(graphs, labels):
        # Move the graph and label to the device
        g = g.to(device)
        l = l.to(device)
        # Forward pass
        logits = model(g)
        loss = F.l1_loss(logits, l.view(-1, 1))
        # Backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()


# Evaluation function
# def evaluate(model, graphs, labels):
#     # Set model to evaluation mode
#     model.eval()
#     # Loop through the graphs and labels and calculate the mean absolute percentage error (MAPE)
#     total_error = 0
#     with torch.no_grad():
#         for g, l in zip(graphs, labels):
#             # Move the graph and label to the device
#             g = g.to(device)
#             l = l.to(device)


class SAGENet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(SAGENet, self).__init__()
        self.conv1 = dgl.nn.SAGEConv(in_size, hidden_size, aggregator_type='mean')
        self.conv2 = dgl.nn.SAGEConv(hidden_size, hidden_size, aggregator_type='mean')
        self.conv3 = dgl.nn.SAGEConv(hidden_size, out_size, aggregator_type='mean')

    def forward(self, g):
        h = g.ndata['weight']
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = self.conv3(g, h)
        return h


# def train(model, data_iter, loss_fn, optimizer, device):
#     model.train()
#     total_loss = 0
#     total_mape = 0
#     for i, (graph, label) in enumerate(data_iter):
#         optimizer.zero_grad()
#         graph = graph.to(device)
#         label = label.to(device)
#         logits = model(graph)
#         loss = loss_fn(logits, label)
#         total_loss += loss.item()
#         mape = torch.mean(torch.abs((label - logits) / label))
#         total_mape += mape.item()
#         loss.backward()
#         optimizer.step()
#     return total_loss / (i + 1), total_mape / (i + 1)


def evaluate(model, graphs, labels):
    model.eval()
    with torch.no_grad():
        logits = model(graphs)
        logits = logits.squeeze()
        labels = labels.float()
        loss = F.mse_loss(logits, labels)
        mape = torch.mean(torch.abs((labels - logits) / labels))
    return loss, mape


# Set the hyperparameters
num_epochs = 100
learning_rate = 0.01
hidden_size = 32

# Set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Load the data
graphs, labels = load_data('/Users/tedlau/PostGraduatePeriod/Graduate/Tasks/Task9-神经网络比较-不会--但又给续上了/data_by_day_simple/datatest.txt',6)

# Preprocess the data
graphs, labels = preprocess_data(graphs, labels)

# Create the model
model = SAGENet(1, hidden_size, 1).to(device)

# Define the optimizer and criterion
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    train_loss, train_mape = train(model, optimizer, graphs, labels)
    val_loss, val_mape = evaluate(model, graphs, labels)

    print(
        f'Epoch {epoch + 1}: train loss = {train_loss:.4f}, train MAPE = {train_mape:.4f}, val loss = {val_loss:.4f}, val MAPE = {val_mape:.4f}')
