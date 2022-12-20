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
            # Make sure the input tensor is contiguous
            h = h.contiguous()
            # Apply the layer
            h = layer(graph, h)
            h = h.contiguous()
            # Apply ReLU activation for hidden layers, except for the last layer
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


    # Create a list of graphs, one for each day
    graphs = []
    labels = []
    for _, group in df.groupby('date'):
        ip_to_id = {}
        id_to_ip = {}
        idx = 0
        for ips in group[['src', 'dst']].values:
            for ip in ips:
                if ip not in ip_to_id:
                    ip_to_id[ip] = idx
                    id_to_ip[idx] = ip
                    idx += 1

        # Convert the src and dst IP addresses to integers using the mapping
        group['src'] = group['src'].apply(lambda x: ip_to_id[x])
        group['dst'] = group['dst'].apply(lambda x: ip_to_id[x])
        # Create a graph for the current day
        g = dgl.DGLGraph()
        # Add the maximum number of nodes to the graph
        g.add_nodes(idx)
        for _, row in group.iterrows():
            src, dst = row['src'], row['dst']
            # Add the source and destination nodes to the graph if they are not already present
            if src >= g.number_of_nodes():
                g.add_nodes(src - g.number_of_nodes() + 1)
            if dst >= g.number_of_nodes():
                g.add_nodes(dst - g.number_of_nodes() + 1)
            # Add an edge to the graph
            g.add_edge(src, dst)
        graphs.append(g)
        labels.append(idx)

    return graphs, labels

    # Define the training function


def train(graphs, labels, window_size, hidden_size, num_layers, aggregator_type, lr, weight_decay, epochs):
    # Create the model
    model = GraphSAGE(hidden_size, hidden_size, num_classes=1, num_layers=num_layers,
                      aggregator_type=aggregator_type)
    model = model.to(device)
    # Define the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    # Create pandas dataframes to store the training and validation loss and MAPE for each epoch
    train_loss = pd.DataFrame(columns=['loss', 'mape'])
    val_loss = pd.DataFrame(columns=['loss', 'mape'])

    # Split the data into training and validation sets
    num_train = int(len(graphs) * 0.8)
    train_graphs = graphs[:num_train]
    train_labels = labels[:num_train]
    val_graphs = graphs[num_train:]
    val_labels = labels[num_train:]

    # Train the model
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        # Loop over the training data in windows
        for i in range(len(train_graphs) - window_size):
            window_graphs = train_graphs[i:i+window_size]
            window_label = train_labels[i+window_size]
            window_graphs = [g.contiguous() for g in window_graphs]
            # Concatenate the graphs in the current window into a single graph
            g = dgl.batch(window_graphs)

            # Concatenate the graphs in the current window into a single graph
            g = dgl.batch(window_graphs)
            # Compute the output of the model on the concatenated graph
            output = model(g)
            # Compute the loss
            loss = loss_fn(output, torch.Tensor([window_label]).to(device))
            total_loss += loss.item()

            # Compute the average loss and MAPE for the epoch
        avg_loss = total_loss / (len(val_graphs) - window_size)
        avg_mape = mape_loss(output.cpu().detach().numpy(), window_label)
        # Add the loss and MAPE for the epoch to the validation dataframe
        val_loss = val_loss.append({'loss': avg_loss, 'mape': avg_mape}, ignore_index=True)

    # Return the model and the training and validation dataframes
    return model, train_loss, val_loss


# Load the data
df = pd.read_csv(
    '/Users/tedlau/PostGraduatePeriod/Graduate/Tasks/Task9-神经网络比较-不会--但又给续上了/data_by_day_simple/datatest.txt',
    sep='	', header=None, names=['src', 'dst', 'weight', 'date'])

# Preprocess the data
graphs, labels = preprocess_data(df, window_size=5)

# Train the model
train(graphs, labels, window_size=5, hidden_size=64, num_layers=2, aggregator_type='mean', lr=0.01,
      weight_decay=0.001, epochs=10)
