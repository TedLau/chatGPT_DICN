import torch
import dgl
import dgl.function as fn
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv
from tqdm import tqdm

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define the GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 hidden_size,
                 num_classes,
                 num_layers,
                 aggregator_type):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(SAGEConv(in_feats, hidden_size, aggregator_type))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(SAGEConv(hidden_size, hidden_size, aggregator_type))
        # output layer
        self.layers.append(SAGEConv(hidden_size, num_classes, aggregator_type))

    def forward(self, graph, features=None):
        # If no node features are provided, use an empty tensor as the node features
        if features is None:
            features = torch.empty((graph.number_of_nodes(), 0)).to(device)
        h = features

        for i, layer in enumerate(self.layers):
            h = h.contiguous()
            h = layer(graph, h)
            h = h.contiguous()
            if i != len(self.layers) - 1:
                h = F.relu(h)
        return h


# Define the mean absolute percentage error (MAPE) loss function
def mape_loss(y_pred, y_true):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


# Define the preprocessing function
def preprocess_data(df, window_size):
    df = df.sort_values(by='date')

    # Create a dictionary to map each node to a unique id
    node_to_id = {}
    id_to_node = {}
    idx = 0
    for nodes in df[['src', 'dst']].values:
        for node in nodes:
            if node not in node_to_id:
                node_to_id[node] = idx
                id_to_node[idx] = node
                idx += 1

    # Convert the src and dst nodes to ids
    df['src'] = df['src'].apply(lambda x: node_to_id[x])
    df['dst'] = df['dst'].apply(lambda x: node_to_id[x])

    # Group the data by date
    groups = df.groupby('date')
    # Create a list of graphs, one for each day
    graphs = []
    for _, group in groups:
        # Create a graph for the current day
        g = dgl.DGLGraph()
        g.add_nodes(len(group))
        for _, row in group.iterrows():
            src, dst = row['src'], row['dst']
            # Add an edge to the graph
            g.add_edge(src, dst)
        graphs.append(g)

    # Compute the labels for each window
    labels = []
    for i in range(window_size, len(graphs)):
        labels.append(graphs[i].number_of_nodes())

    return graphs, labels

    # Define the training function


def train(graphs, labels, window_size, hidden_size, num_layers, aggregator_type, lr, weight_decay, epochs):
    # Create the model
    model = GraphSAGE(hidden_size, hidden_size, num_classes=1, num_layers=num_layers,
                      aggregator_type=aggregator_type)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Train the model
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i in range(len(graphs) - window_size):
            # Create a subgraph for the current window
            subgraph = dgl.batch(graphs[i:i + window_size])
            # Compute the node features for the subgraph
            node_features = None
            # Compute the logits for the subgraph
            logits = model(subgraph, None)

            # Compute the loss for the current window
            loss = mape_loss(logits, labels[i])
            total_loss += loss.item()
            # Backpropagate the error
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f'Epoch {epoch + 1}: loss = {total_loss / len(graphs)}')


# Load the data
df = pd.read_csv(
    '/Users/tedlau/PostGraduatePeriod/Graduate/Tasks/Task9-神经网络比较-不会--但又给续上了/data_by_day_simple/datatest.txt',
    sep='	', header=None, names=['src', 'dst', 'weight', 'date'])

# Preprocess the data
graphs, labels = preprocess_data(df, window_size=6)

# Train the model
train(graphs, labels, window_size=6, hidden_size=64, num_layers=2, aggregator_type='mean', lr=0.01,
      weight_decay=0.001, epochs=10)
