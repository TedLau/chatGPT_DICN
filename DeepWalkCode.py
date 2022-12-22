import torch
import dgl
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        # Add the edges to the graph
        for src, dst in group[['src', 'dst']].values:
            g.add_edge(src, dst)
        # Add the graph to the list of graphs
        graphs.append(g)
        # Calculate the number of nodes in the graph
        node_number = g.number_of_nodes()
        # Add the node number to the list of labels
        labels.append(node_number)
        # Add the node number to the list of previous node numbers
        previous_node_numbers.append(node_number)

    for i in range(window_size - 1, len(previous_node_numbers)):
        # Add the node numbers from the previous days to the list of labels
        labels.append(previous_node_numbers[i - window_size + 1:i + 1])
    # Convert the lists to numpy arrays and return them
    return np.array(graphs), np.array(labels)


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
        # self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 64)
        # self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 1)

    def forward(self, node_sequence):
        # node_embeddings = self.node_embeddings(node_sequence,dim=1)
        node_embeddings = node_embeddings.mean(dim=1)
        out = self.linear1(node_embeddings)
        out = F.relu(out)
        # out = self.linear2(out)
        # out = F.relu(out)
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
        if len(neighbors) > 0:
            next_node = np.random.choice(neighbors)
        else:

            next_node = np.random.choice(graph.nodes())
        walk.append(next_node)
        current_node = next_node
    return walk
df = pd.read_csv(
    '/Users/tedlau/PostGraduatePeriod/Graduate/Tasks/Task9-神经网络比较-不会--但又给续上了/data_by_day_simple/datatest.txt',
    sep='	', header=None, names=['src', 'dst', 'weight', 'date'])
graphs, labels = preprocess_data(df, window_size=5)
# Define the model parameters
num_nodes = len(ip_to_id)  # Total number of unique nodes in the data
embedding_dim = 0  # Embedding dimension for the node embeddings
window_size = 5  # Window size for the random walk
train_graphs, test_graphs = graphs[:-1], graphs[-1:]
train_labels, test_labels = labels[:-1], labels[-1:]

# Initialize the model
model = DeepWalk(num_nodes, embedding_dim, window_size)
model.to(device)  # Move the model to the designated device (CPU or GPU)

# Define the optimizer and the learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
# Training loop
for epoch in range(num_epochs):
  for i, (graph, label) in enumerate(zip(graphs, labels)):
    # Convert the graph and label to tensors and move them to the designated device
    graph = graph.to(device)
    # label = label.to(device)

    # Reset the gradients
    optimizer.zero_grad()

    # Generate a random walk for the current graph
    walk = random_walk(graph, np.random.choice(graph.nodes()), window_size)

    # Pass the random walk through the model to get a prediction
    prediction = model(torch.tensor(walk).to(device))

    # Calculate the loss
    loss = mape_loss(prediction, label)

    # Backpropagate the loss and update the model weights
    loss.backward()
    optimizer.step()

# Prediction loop
predictions = []
for i, graph in enumerate(test_graphs):
  # Generate a random walk for the current graph
  walk = random_walk(graph, np.random.choice(graph.nodes()), window_size)

  # Pass the random walk through the model to get a prediction
  prediction = model(torch.tensor(walk).to(device))

  # Append the prediction to the list of predictions
  predictions.append(prediction.item())
