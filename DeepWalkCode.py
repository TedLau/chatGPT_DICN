import torch
import dgl
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def mape_loss(y_pred, y_true):
#     y_true, y_pred = torch.Tensor(y_true).detach(), torch.Tensor(y_pred).detach()
#
#     return torch.mean(torch.abs((y_true - y_pred) / y_true))
def mape_loss(y_pred, y_true):
    y_pred = y_pred.float()
    y_true = y_true
    return torch.mean(torch.abs((y_true - y_pred) / y_true))

    # Define the DeepWalk model


class DeepWalk(nn.Module):
    def __init__(self, num_nodes, embedding_dim, window_size):
        super(DeepWalk, self).__init__()
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.num_nodes = num_nodes
        self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 1)

    def forward(self, node_sequence):
        node_embeddings = self.node_embeddings(node_sequence)
        node_embeddings = node_embeddings.mean(dim=1)
        out = self.linear1(node_embeddings)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        return out


# Define the mean absolute percentage error (MAPE) loss function

ip_to_id = {}
def random_walk(graph, start_node, walk_length):
    """
    Generates a random walk on the given graph starting from the given node.

    Parameters:
        graph (dgl.DGLGraph): The graph to perform the random walk on.
        start_node (int): The node to start the random walk from.
        walk_length (int): The length of the random walk.

    Returns:
        list: A list of nodes representing the random walk.
    """
    walk = [start_node]
    current_node = start_node
    for _ in range(walk_length):
        # Get the neighbors of the current node
        neighbors = graph.successors(current_node)
        # Choose a random neighbor
        next_node = np.random.choice(neighbors)
        walk.append(next_node)
        current_node = next_node
    return walk


# Define the preprocessing function
def preprocess_data(df, window_size):
    df = df.sort_values(by='date')

    # Create a list of graphs, one for each day
    graphs = []
    labels = []
    previous_node_numbers = []
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

        node_sequence = torch.tensor(list(range(g.number_of_nodes())), dtype=torch.long)
        g.ndata['node_sequence'] = node_sequence
        previous_node_numbers.append(node_sequence)
        previous_node_numbers = list(reversed(previous_node_numbers))
        node_sequence = torch.cat(previous_node_numbers, dim=0)
        graphs.append(node_sequence)
        labels.append(g.number_of_nodes())
        # Move to the next day

    # Return the list of graphs and labels
    return graphs, labels


# Load the data
df = pd.read_csv(
    '/Users/tedlau/PostGraduatePeriod/Graduate/Tasks/Task9-神经网络比较-不会--但又给续上了/data_by_day_simple/datatest.txt',
    sep='	', header=None, names=['src', 'dst', 'weight', 'date'])
# Load the data

window_size = 7
# Preprocess the data
graphs, labels = preprocess_data(df, window_size=7)

# Initialize the DeepWalk model
model = DeepWalk(num_nodes=df['src'].nunique(), embedding_dim=64, window_size=7).to(device)

# Set the optimizer and criterion
optimizer = torch.optim.Adam(model.parameters())
criterion = mape_loss

# Set the number of epochs and the learning rate
num_epochs = 100
learning_rate = 0.001
# Train the model
# model = DeepWalk(num_nodes=num_nodes, embedding_dim=16, window_size=window_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(200):
    total_loss = 0
    for i, (graph, label) in enumerate(zip(graphs, labels)):
        # Get the node sequences for the current graph
        node_sequences = []
        for j in range(window_size):
            if i - j < 0:
                break
            node_sequences.append(graphs[i - j])#.ndata['node_sequence']
        # Reverse the list of node sequences so that the most recent graph is first
        node_sequences = list(reversed(node_sequences))
        node_sequences[0] = node_sequences[0][0:64]
        for i in range(len(node_sequences)):
            node_sequences[i] = node_sequences[i][i*64:i*64+64]
        # Concatenate the node sequences into a single tensor
        node_sequence = torch.cat(node_sequences, dim=0)
        # Make the prediction
        prediction = model(node_sequence)
        # Compute the loss
        loss = mape_loss(prediction, label)
        total_loss += loss.item()
        # Zero the gradients
        optimizer.zero_grad()
        # Backpropagate the loss
        # loss.backward()
        # Update the model weights
        optimizer.step()
    if epoch % 20 == 0:
        print(f'Epoch {epoch}, loss {total_loss / len(labels)}')

#


# # Preprocess the data
# node_lists, labels = preprocess_data(df, window_size=1)
#
# # Convert the labels to a tensor
# labels = torch.tensor(labels).to(device)
#
# # Create the model
# model = DeepWalk(num_nodes=len(ip_to_id)).to(device)
#
# # Define the optimizer and criterion
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.MSELoss()
#
# # Training loop
# for epoch in range(5):
#     total_loss = 0
#     for node_list, label in zip(node_lists, labels):
#         # Convert the node list to a tensor
#         node_list = torch.tensor(node_list).to(device)
#         # Forward pass
#         output = model(None, node_list)
#         output = output.view(1, -1)
#
#         output = output.float()
#         # Compute the loss
#         loss = mape_loss(output, labels)
#
#         total_loss += loss.item()
#         # Zero the gradients
#         optimizer.zero_grad()
#         # Backward pass
#         loss.backward()
#         # Update the parameters
#         optimizer.step()
#     print(f'Epoch {epoch + 1}: Loss = {total_loss / len(node_lists)}')
#
# # Evaluation loop
# total_mape = 0
# for node_list, label in zip(node_lists, labels):
#     # Convert the node list to a tensor
#     node_list = torch.tensor(node_list).to(device)
#     # Forward pass
#     output = model(None, node_list)
#     # Compute the mean absolute percentage error
#     mape = mape_loss(output.item(), label.item())
#     total_mape += mape
# print(f'Mean absolute percentage error: {total_mape / len(node_lists)}')
