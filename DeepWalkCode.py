import os

import torch
import dgl
import numpy as np
import pandas as pd
import pickle
import torch.nn as nn
from dgl import optim
from tqdm import tqdm
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv,GATConv
# from python.dgl.sampling import random_walk
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from dgl.nn.pytorch import
df = pd.read_csv(
    '/Users/tedlau/PostGraduatePeriod/Graduate/Tasks/Task9-神经网络比较-不会--但又给续上了/data_by_day_simple/datatest.txt',
    sep='	', header=None, names=['src', 'dst', 'weight', 'date'])
# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import dgl
import torch.nn as nn


class DeepWalk(nn.Module):
    def __init__(self, num_nodes, embedding_dim, dropout_rate=0.5):
        super(DeepWalk, self).__init__()
        self.num_layers = 1
        self.layers = nn.ModuleList()
        # self.layers.append(dgl.nn.GraphConv(embedding_dim, embedding_dim))
        self.layers.append(dgl.nn.GraphConv(embedding_dim, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)
        # self.window_size = window_size

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h

class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 hidden_size,
                 num_classes,
                 # num_layers,
                 ):
        # in_feats = 0
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(SAGEConv(in_feats, hidden_size, aggregator_type='lstm'))
        # hidden layers

        # output layer
        self.layers.append(SAGEConv(hidden_size, 1, aggregator_type='lstm'))

        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h
class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, dropout=0.5):
        super(GAT, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p=dropout)

        self.layers = nn.ModuleList([
            GATConv(in_dim, hidden_dim, num_heads, dropout),
            GATConv(hidden_dim * num_heads, out_dim, num_heads, dropout),
        ])

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h
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


# Initialize the list
node_numbers = []

# Group the data by date
for _, group in df.groupby('date'):
    # Count the number of unique nodes
    node_number = group['src'].nunique() + group['dst'].nunique()
    # Append the node number to the list
    node_numbers.append(node_number)


def pre():
    embedding_dim = 1
    window_size = 5
    learning_rate = 0.001
    num_epochs = 20
    # Initialize the model and optimizer
    sum = 0
    for n in node_numbers:
        sum += n
    # sum = [sum += n for n in node_numbers]
    model = DeepWalk(sum, embedding_dim, window_size).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # Initialize the list
    graphs = []


def mape_loss(y_pred, y_true):
    y_pred = y_pred.float()
    y_true = y_true.float()
    return torch.mean(torch.abs((y_true - y_pred) / y_true))


# Group the data by date
def depre():
    for _, group in df.groupby('date'):

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


days = ['20220902',
        '20220903',
        '20220904',
        '20220905',
        '20220906',
        '20220907',
        '20220908',
        '20220909',
        '20220910',
        '20220911',
        '20220912',
        '20220913']
day_labels = [141, 90, 100, 143, 124, 190, 155, 270, 397, 359]


def make_few_days_graph(start_date, cnt):
    i = 0  # 弄结束
    j = 0  # 弄开始
    g = dgl.DGLGraph()
    for _, group in df.groupby('date'):

        if j < start_date - 1:
            j += 1
            continue
        i += 1
        if i > cnt:
            break
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

        # Add the maximum number of nodes to the graph
        g.add_nodes(idx)
        g = dgl.add_self_loop(g)
        # Add the edges to the graph
        for src, dst in group[['src', 'dst']].values:
            g.add_edge(src, dst)
        g = dgl.add_self_loop(g)
    node_feature = [1 for _ in range(128)]
    features = [node_feature for _ in range(g.num_nodes())]
    features = torch.tensor(features).float().to(device)
    label = [day_labels[start_date + cnt - 1]]
    labels = [label for _ in range(g.num_nodes())]
    labels = torch.tensor(labels).float().to(device)
    return [[g, features], labels]


# graph1,feat1,label1 = make_few_days_graph(2,2)
# graph2,feat2,label2 = make_few_days_graph(1,2)
def new_train(model, train_set):
    # define train/val samples, loss function and optimizer
    # loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    loss_hist = []
    # training loop
    for epoch in range(300):
        model.train()
        loss_single = 0
        for batch in train_set:
            input_g = batch[0][0]
            input_f = batch[0][1]
            labels = batch[1]
            logits = model(input_g, input_f)
            loss = loss_fcn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_single += loss.item()
            loss_hist.append(loss_single)
        # acc = evaluate(g, features, labels, val_mask, model)
        print("Epoch {:05d} | Loss {:.4f}"
              .format(epoch, loss_single))
    # save_model(model)
    # save_pkl(loss_hist, "loss", f"{get_model_number()}")
def evaluate(model, val_set):
    model.eval()
    with torch.no_grad():
        input_g = val_set[0][0]
        input_f = val_set[0][1]
        labels = val_set[1]
        logits = model(input_g, input_f)
        test_metricses = []
        r2_scores = []
        if model.__class__.__name__ == 'GAT':
            logits = logits.squeeze(dim=1).squeeze(dim=2)
        mape_score = mean_absolute_percentage_error(labels.cpu().detach().numpy(), logits.cpu().detach().numpy())
        r2_score_single = r2_score(labels.cpu().detach().numpy(), logits.cpu().detach().numpy())
        test_metricses.append(mape_score)
        r2_scores.append(r2_score_single)

        print(
            f"Output status: min {logits.cpu().detach().numpy().min()}, max {logits.cpu().detach().numpy().max()}, mean {logits.cpu().detach().numpy().mean()}")

        mape_score_test = np.mean(test_metricses)
        r2_score_single_test = np.mean(r2_scores)
        torch.cuda.empty_cache()
        return mape_score_test, r2_score_single_test


model_path = "output/models"


def get_model_number():
    names = os.listdir(model_path)
    tot = 0
    for name in names:
        if name.startswith("saved_model"):
            tot += 1
    return tot


def save_model(model):
    model_number = get_model_number()
    torch.save(model.state_dict(), f"{model_path}/saved_model{model_number + 1}")
def save_pkl(dictionnary, directory, file_name):
    """Save the dictionnary in the directory under the file_name with pickle"""
    with open(f"output/{directory}/{file_name}.pkl", "wb") as output:
        pickle.dump(dictionnary, output)


def load_model(model_number=0):
    if model_number == 0:
        model_number = get_model_number()
        return torch.load(f"{model_path}/saved_model{model_number}")
    else:
        return torch.load(f"{model_path}/saved_model{model_number}")


if __name__ == '__main__':
    in_feat = 128
    out_size = 1
    # model = DeepWalk(in_feat, 128, out_size).to(device)
    # model = GraphSAGE(in_feat, 128, out_size).to(device)
    model = GAT(in_feat,128,1,out_size)
    tain_set2 = [make_few_days_graph(4,2)]
    train_set = [make_few_days_graph(1, 6),
                 make_few_days_graph(2, 6),
                 make_few_days_graph(3, 6),
]
    new_train(model, train_set)
    # model.load_state_dict(load_model(0))
    test_set = make_few_days_graph(4, 6)
    # test the model
    print('Testing...')
    mape, r2 = evaluate(model, test_set)
    print("Test mape {:.4f}, r2 {:.4f}".format(mape, r2))


def generate_emb():
    # Initialize the lists
    cascade_embeddings = []
    labels = []

    # Loop over the graphs
    for g, label in tqdm(zip(graphs, node_numbers)):
        # Convert the graph to a DGLGraph
        # g = dgl.DGLGraph(g)
        # Initialize the node embeddings
        node_embeddings = torch.zeros(g.number_of_nodes(), embedding_dim)
        # Loop over the nodes
        for node in tqdm(range(g.number_of_nodes())):
            # Generate a random walk for the node
            walk = random_walk(g, node, 5)
            # Convert the walk to a tensor
            walk = torch.tensor(walk).to(device)
            # Get the node embedding
            node_embedding = model.node_embeddings(walk)
            # Average the node embedding
            node_embedding = node_embedding.mean(dim=0)
            # Update the node embeddings
            node_embeddings[node] = node_embedding
        # Average the node embeddings to get the cascade embedding
        cascade_embedding = node_embeddings.mean(dim=0)
        # Append the cascade embedding and label to the lists
        cascade_embeddings.append(cascade_embedding)
        labels.append(label)


def preprocess_data(cascade_embeddings, labels):
    features = []
    for i in range(len(labels)):
        # Get the cascade embeddings for the previous days
        prev_embeddings = cascade_embeddings[max(0, i - window_size):i]
        # Pad the list with zeros if there are not enough embeddings
        while len(prev_embeddings) < window_size:
            prev_embeddings.insert(0, np.zeros(embedding_dim))
        # Stack the embeddings into a single tensor
        # Convert the prev_embeddings list to a list of tensors
        new_pr = []
        for x in prev_embeddings:
            if torch.is_tensor(x):
                new_pr.append(x)
                continue
            else:
                new_pr.append(torch.from_numpy(x))
        # prev_embeddings = [torch.from_numpy(x) if !torch.is_tensor(x) for x in prev_embeddings]

        # Stack the tensors to create a single feature tensor
        feature = torch.stack(new_pr)

        features.append(feature)
    return features, labels


