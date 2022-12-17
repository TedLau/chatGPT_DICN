import torch
import dgl
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F


def preprocess_data(df, window_size):
    # Sort the dataframe by date
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
    df['weight'] = df['weight'].replace('None', np.nan)
    df['weight'] = df['weight'].astype(float)
    mean_weight = df['weight'].mean()
    df['weight'] = df['weight'].fillna(mean_weight)
    # Create a list of graphs, one for each day
    labels = []
    graphs = []
    for _, group in groups:
        # Create a graph for the current day
        g = dgl.DGLGraph()

        g.add_nodes(len(set(group.src) | set(group.dst)))
        labels.append(len(set(group.src) | set(group.dst)))
        for _, row in group.iterrows():
            src, dst, weight = row['src'].astype(int), row['dst'].astype(int), row['weight']
            # Add an edge to the graph
            # length = len(set(s))
            g.add_edge(src, dst)  # , weight=weight
        graphs.append(g)

    # Compute the labels for each window
    #
    # # for i in range(window_size, len(graphs)):
    # #     labels.append(len(np.unique(graphs[i].ndata['id'])))
    # for i in range(len(graphs)):
    #     labels.append(i.num_nodes)

    return graphs, labels


def train(graphs, labels, model, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for g, label in zip(graphs, labels):
        # Convert the graph to a tensor and send it to the device
        g = g.to(device)
        # Generate node embeddings for the graph
        node_embeddings = model(g)
        # Compute the prediction
        prediction = node_embeddings.sum()
        # Compute the loss
        loss = criterion(prediction, label)
        # Backpropagate the error and update the model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(graphs)


def evaluate(graphs, labels, model, criterion, device):
    model.eval()
    total_loss = 0
    for g, label in zip(graphs, labels):
        # Convert the graph to a tensor and send it to the device
        g = g.to(device)
        # Generate node embeddings for the graph
        node_embeddings = model(g)
        # Compute the prediction
        prediction = node_embeddings.sum()
        # Compute the loss
        loss = criterion(prediction, label)
        total_loss += loss.item()
    return total_loss / len(graphs)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


class NeuralNet(nn.Module):
    def __init__(self, num_nodes, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(num_nodes, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, graph):
        node_embeddings = self.deepwalk(graph)
        x = self.fc1(node_embeddings)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def deepwalk(self, graph):
        # Perform the DeepWalk algorithm on the graph to generate node embeddings
        # ...
        return node_embeddings


def main():
    # Load the data
    df = pd.read_csv(
        '/Users/tedlau/PostGraduatePeriod/Graduate/Tasks/Task9-神经网络比较-不会--但又给续上了/data_by_day_simple/datatest.txt',
        sep='	', header=None, names=['src', 'dst', 'weight', 'date'])

    # Set the hyperparameters
    window_size = 6
    hidden_size = 64
    num_layers = 1
    lr = 0.001
    weight_decay = 0.0005
    epochs = 100

    # Preprocess the data
    graphs, labels = preprocess_data(df, window_size)
    num_nodes = len(set(df.src) | set(df.dst))
    num_classes = 1

    # Split the data into training and test sets
    train_graphs = graphs[:-window_size]
    train_labels = labels[:-window_size]
    test_graphs = graphs[-window_size:]
    test_labels = labels[-window_size:]

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model and optimizer
    model = NeuralNet(num_nodes, hidden_size, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Train the model
    for epoch in range(epochs):
        train_loss = train(train_graphs, train_labels, model, optimizer, criterion, device)
        test_loss = evaluate(test_graphs, test_labels, model, criterion, device)
        print(f'Epoch {epoch + 1}: train loss = {train_loss:.4f}, test loss = {test_loss:.4f}')

    # Make predictions on the test set
    predictions = []
    for g in test_graphs:
        # Convert the graph to a tensor and send it to the device
        g = g.to(device)
        # Generate node embeddings for the graph
        node_embeddings = model(g)
        # Compute the prediction
        prediction = node_embeddings.sum()
        predictions.append(prediction.item())

    # Compute the MAPE
    mape = mean_absolute_percentage_error(test_labels, predictions)
    print(f'MAPE = {mape:.2f}%')


if __name__ == '__main__':
    main()
