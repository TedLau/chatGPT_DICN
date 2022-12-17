import dgl
import dgl.function as fn
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def load_data(file_path, window_size):
    df = pd.read_csv(file_path, sep='	', names=['src', 'dst', 'weight', 'timestamp'])
    df, node_dict = preprocess_data(df)
    graphs = []
    for i in tqdm(range(len(df) - window_size)):
        g = dgl.DGLGraph()
        g.add_nodes(len(node_dict))
        src = df.iloc[i:i+window_size]['src'].values
        dst = df.iloc[i:i+window_size]['dst'].values
        weight = df.iloc[i:i+window_size]['weight'].values
        g.add_edges(src, dst, {'weight': weight})
        graphs.append(g)
    labels = df.iloc[window_size:]['timestamp'].apply(lambda x: len(df[df['timestamp'] == x]['src'].unique())).values
    return graphs, labels#, node_dict


def preprocess_data(df):
    df['src'] = df['src'].astype(str)
    df['dst'] = df['dst'].astype(str)
    df['timestamp'] = df['timestamp'].astype(str)
    unique_nodes = set(df['src'].unique()).union(df['dst'].unique())
    node_dict = {node: i for i, node in enumerate(unique_nodes)}
    df['src'] = df['src'].map(node_dict)
    df['dst'] = df['dst'].map(node_dict)
    df['weight'] = df['weight'].replace('None', np.nan)
    df['weight'] = df['weight'].astype(float)
    mean_weight = df['weight'].mean()
    df['weight'] = df['weight'].fillna(mean_weight)
    return df, node_dict


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


def train(model, data_iter, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    total_mape = 0
    for i, (graph, label) in enumerate(data_iter):
        optimizer.zero_grad()
        graph = graph.to(device)
        label = label.to(device)
        logits = model(graph)
        loss = loss_fn(logits, label)
        total_loss += loss.item()
        mape = torch.mean(torch.abs((label - logits) / label))
        total_mape += mape.item()
        loss.backward()
        optimizer.step()
    return total_loss / (i + 1), total_mape / (i + 1)


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
