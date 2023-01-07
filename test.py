# %%
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import pickle
from sklearn import svm
import numpy as np
import scipy.sparse
from dgl.nn.pytorch import SAGEConv, GATConv
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
# from dgl import AddSelfLoop
import argparse
import copy
import networkx as nx
# from keras.layers import Dense
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
import gc
from gensim.models import Word2Vec

# from python.dgl.nn.pytorch import Sequential

out_put_size = 1


def save_pkl(dictionnary, directory, file_name):
    """Save the dictionnary in the directory under the file_name with pickle"""
    with open(f"output/{directory}/{file_name}.pkl", "wb") as output:
        pickle.dump(dictionnary, output)


# %%
import platform

sysstr = platform.system()
if sysstr == "Linux":
    # paths
    data_path = "/home/liupw/data/"
elif sysstr == "Darwin":
    # paths
    data_path = "/Users/tedlau/PostGraduatePeriod/Graduate/Tasks/Task9-神经网络比较-不会--但又给续上了/data_with_label/"
else:
    data_path = "/home/liupw/data/"

with open(f"{data_path}data_dict_sparse_update.pkl", "rb") as f:
    data_dict_sparse = pickle.load(f)

with open(f"{data_path}matrix_data_have_node.pkl", "rb") as f:
    matrix_data_have_node = pickle.load(f)

with open(f"{data_path}ipaddress_dict.pkl", "rb") as f:
    ipaddress_dict = pickle.load(f)

with open(f"{data_path}matrix_label.pkl", "rb") as f:
    matrix_label = pickle.load(f)


def make_graph(start: int, end: int, device):
    rows = []
    cols = []
    values = []
    for i in range(start, end + 1):
        rows.extend(data_dict_sparse[str(i)][0])
        cols.extend(data_dict_sparse[str(i)][1])
        values.extend(data_dict_sparse[str(i)][2])

    # make the graph no direction
    tmp_rows = copy.deepcopy(rows)
    rows.extend(cols)
    cols.extend(tmp_rows)
    values.extend(values)
    del tmp_rows

    rows = np.array(rows)
    cols = np.array(cols)
    values = np.array(values)
    sparseM_v = scipy.sparse.coo_matrix((values, (rows, cols)))

    # label_position = 0
    # for idx, date in enumerate(matrix_data_have_node[0]):
    #     if date > label_num:
    #         label_position = idx
    #         break
    # rows_date  = np.array(matrix_data_have_node[0][:label_position]) - 1
    # cols_date  = np.array(matrix_data_have_node[1][:label_position])
    # values_date = np.array(matrix_data_have_node[2][:label_position])

    # sparseM_date = scipy.sparse.coo_matrix((values_date, (rows_date, cols_date)))
    G = dgl.from_scipy(sparseM_v)
    G = dgl.add_self_loop(G)
    # G = transform(G)
    print(G)

    G = G.to(device)
    nodes_number = G.num_nodes()

    feature = [1 for _ in range(128)]
    features = [feature for _ in range(nodes_number)]
    features = torch.tensor(features).float().to(device)
    labels_tot_num = matrix_label[1][end: end + 1]
    labels = [labels_tot_num for _ in range(nodes_number)]
    labels = torch.tensor(labels).float().to(device)
    return [[G, features], labels]


# class DeepWalk(nn.Module):
#     def __init__(self, num_nodes, embedding_dim, dropout_rate=0.5):
#         super(DeepWalk, self).__init__()
#         self.num_layers = 1
#         self.layers = nn.ModuleList()
#         # self.layers.append(dgl.nn.GraphConv(embedding_dim, embedding_dim))
#         self.layers.append(dgl.nn.GraphConv(embedding_dim, 1))
#         self.dropout = nn.Dropout(dropout_rate)
#         self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)
#         self.window_size = 15
#
#     def forward(self, g, features):
#         h = features
#         for i, layer in enumerate(self.layers):
#             if i != 0:
#                 h = self.dropout(h)
#             h = layer(g, h)
#         return h
import random
import numpy as np


class DeepWalk:
    def __init__(self, graph, window_size, walk_length, num_walks, embedding_dim):
        self.graph = graph
        self.window_size = window_size
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.embedding_dim = embedding_dim

    def generate_walks(self):
        """
        Generate random walks on the graph.
        """
        walks = []
        for _ in range(self.num_walks):
            walk = [random.choice(list(self.graph.nodes()))]
            while len(walk) < self.walk_length:
                cur_node = walk[-1]
                neighbors = graph.successors(cur_node)
                if len(neighbors) > 0:
                    walk.append(random.choice(neighbors))
                else:
                    break
            walks.append(walk)
        return walks

    def get_contexts(self, walk):
        """
        Get contexts for each node in the walk.
        """
        contexts = []
        for i, node in enumerate(walk):
            left = max(0, i - self.window_size)
            right = min(len(walk), i + self.window_size + 1)
            contexts.append([walk[j] for j in range(left, right) if j != i])
        return contexts

    def train(self, walks):
        """
        Train the word2vec model using the generated walks.
        """
        model = Word2Vec(walks, vector_size=self.embedding_dim, window=self.window_size, min_count=0, sg=1, workers=4,
                         compute_loss=True)
        print("Loss: ", model.get_latest_training_loss())
        return model

    def get_node_embeddings(self):
        """
        Get node embeddings for all nodes in the graph.
        """
        walks = self.generate_walks()
        contexts = []
        for walk in walks:
            contexts.extend(self.get_contexts(walk))
        model = self.train(contexts)
        node_embeddings = {}
        i = 0
        index = []
        for node in self.graph.nodes():
            node = node.item()
            if node in model.wv:
                index.append(node)
                node_embeddings[node] = node_embeddings.get(node, model.wv[node])
            else:
                i += 1
                # print('Motherfucker ',node)
        print('total cnt:', i)
        print(index)
        model.save("./deepwalk_embeddings.bin")

        return node_embeddings


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


class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(dglnn.GraphConv(in_size, hid_size, activation=F.relu))
        self.layers.append(dglnn.GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h


# Define a PyTorch neural network for regression
class RegressionNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def evaluate(model, val_set):
    model.eval()
    with torch.no_grad():
        input_g = val_set[0][0]
        input_f = val_set[0][1]
        labels = val_set[1]
        logits = model(input_g, input_f)
        test_metricses = []
        r2_scores = []

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


def evaluate_D(model, val_set):
    model.eval()
    with torch.no_grad():
        x_train = val_set[1]
        # x_train = np.concatenate((batch[1],batch[0]),axis=1)
        x_train = np.array(list(x_train.values()))

        x_train = np.expand_dims(x_train, axis=1)
        x_train = torch.from_numpy(x_train).float()
        x_train = x_train.to(device)

        # y_train = val_set[2]
        # y_train = y_train.to(device)
        logits = model(x_train)
        # input_g = val_set[0][0]
        # input_f = val_set[0][1]
        labels = val_set[2]
        # logits = model(input_g, input_f)
        test_metricses = []
        r2_scores = []
        logits = logits.squeeze()
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


def evaluate_M(model, val_set):
    model.eval()
    with torch.no_grad():
        x_train = val_set[1]
        # x_train = batch[1]
        x_train = x_train.split()
        x_train = [float(value) for value in x_train]

        # x_train = np.concatenate((batch[1],batch[0]),axis=1)
        x_train = np.array(x_train)
        x_train = x_train.astype('float')  # numpy强制类型转换

        # x_train = np.array(x_train)

        x_train = torch.from_numpy(x_train).float()
        x_train = x_train.to(device)

        # y_train = val_set[2]
        # y_train = y_train.to(device)
        logits = model(x_train)
        logits = logits.repeat(6)
        # input_g = val_set[0][0]
        # input_f = val_set[0][1]
        labels = val_set[2]
        print(labels)
        print(logits)
        # logits = model(input_g, input_f)
        test_metricses = []
        r2_scores = []
        # logits = logits.squeeze()
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


def save_pkl(dictionnary, directory, file_name):
    """Save the dictionnary in the directory under the file_name with pickle"""
    with open(f"output/{directory}/{file_name}.pkl", "wb") as output:
        pickle.dump(dictionnary, output)


def train(model, train_set):
    torch.cuda.empty_cache()
    gc.collect()
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
    save_model(model)
    save_pkl(loss_hist, "loss", f"{get_model_number()}")


# model_path = "/home/liupw/dgl-file/py_files/output/GraphSAGE_model"
# model_path = "/home/liupw/dgl-file/py_files/output/metapath2vec_model"
model_path = "/home/liupw/dgl-file/py_files/output/DeepWalk_model"

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
    # with open(f"{model_path}/saved_model{model_number + 1}.txt",'a+') as w:
    #     w.writelines(model.__class__.__name__ )


def load_model(model_number=0):
    if model_number == 0:
        model_number = get_model_number()
        return torch.load(f"{model_path}/saved_model{model_number}")
    else:
        return torch.load(f"{model_path}/saved_model{model_number}")


train_status = True
day_labels = [5408,
              10623,
              13050,
              21746,
              390949,
              724266,
              1082675,
              1566559]


# Define the LSTM model
# Define the LSTM model
# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input_tensor):
        # Run the input through the LSTM
        output = self.lstm(input_tensor)

        # Apply the linear layer
        output = self.linear(output)

        return output


def make_features(node_embeddings, start_date):
    batch_embeddings = node_embeddings[start_date - 1]
    nodes_number = 200

    feature = [1 for _ in range(128)]
    features = [feature for _ in range(nodes_number)]
    features = torch.tensor(features).float().to(device)
    labels_tot_num = day_labels[start_date]
    labels = [labels_tot_num for _ in range(nodes_number)]
    labels = torch.tensor(labels).float().to(device)
    return [features, batch_embeddings, labels]


# Define the model architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def DeepWalk_train(model, train_set):
    # model = RegressionNet(input_size=128, hidden_size=64, output_size=1)
    # Load the node embeddings file into a PyTorch tensor
    # node_embeddings = torch.tensor(node_embeddings_file)
    torch.cuda.empty_cache()
    gc.collect()
    # define train/val samples, loss function and optimizer
    # loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    loss_hist = []
    # training loop
    # epoch = 0
    l2_penalty = 0.01
    for epoch in range(10000):
        # epoch += 1
        # model.train()
        loss_single = 0
        for batch in train_set:
            x_train = batch[1]
            # x_train = np.concatenate((batch[1],batch[0]),axis=1)
            x_train = np.array(list(x_train.values()))

            x_train = np.expand_dims(x_train, axis=1)
            x_train = torch.from_numpy(x_train).float()
            x_train = x_train.to(device)

            y_train = batch[2]
            y_train = y_train.to(device)
            y_pred = model(x_train)
            y_pred = y_pred.squeeze()
            # Compute the loss
            loss = loss_fcn(y_pred, y_train)
            l2_loss = 0
            for param in model.parameters():
                l2_loss += param.norm(2)
            l2_loss *= l2_penalty
            # Zero the gradients
            optimizer.zero_grad()
            loss += l2_loss
            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()
            loss_single += loss.item()
            loss_hist.append(loss_single)
        # acc = evaluate(g, features, labels, val_mask, model)
        print("Epoch {:05d} | Loss {:.4f}"
              .format(epoch, loss_single))
    save_model(model)
    save_pkl(loss_hist, "loss", f"{get_model_number()}")


def metapath2vec_train(model, train_set):
    # model = RegressionNet(input_size=128, hidden_size=64, output_size=1)
    # Load the node embeddings file into a PyTorch tensor
    # node_embeddings = torch.tensor(node_embeddings_file)
    torch.cuda.empty_cache()
    gc.collect()
    # define train/val samples, loss function and optimizer
    # loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    loss_hist = []
    # training loop
    # epoch = 0
    for epoch in range(439):
        # epoch += 1
        # model.train()
        loss_single = 0
        for batch in train_set:
            x_train = batch[1]
            x_train = x_train.split()
            x_train = [float(value) for value in x_train]

            # x_train = np.concatenate((batch[1],batch[0]),axis=1)
            x_train = np.array(x_train)
            x_train = x_train.astype('float')  # numpy强制类型转换

            x_train = np.array(x_train)

            # x_train = np.expand_dims(x_train, axis=1)
            x_train = torch.from_numpy(x_train).float()
            x_train = x_train.to(device)

            y_train = batch[2]
            y_train = y_train.to(device)
            y_pred = model(x_train)
            y_pred = y_pred.squeeze()
            # Compute the loss
            loss = loss_fcn(y_pred, y_train)

            # Zero the gradients
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()
            loss_single += loss.item()
            loss_hist.append(loss_single)
        # acc = evaluate(g, features, labels, val_mask, model)
        print("Epoch {:05d} | Loss {:.4f}"
              .format(epoch, loss_single))
    save_model(model)
    save_pkl(loss_hist, "loss", f"{get_model_number()}")


def make_meta_embeddings(node_embeddings, start_date):
    batch_embeddings = node_embeddings[start_date - 1]
    nodes_number = 6

    feature = [1 for _ in range(128)]
    features = [feature for _ in range(nodes_number)]
    features = torch.tensor(features).float().to(device)
    labels_tot_num = day_labels[start_date]
    labels = [labels_tot_num for _ in range(nodes_number)]
    labels = torch.tensor(labels).float().to(device)
    return [features, batch_embeddings, labels]


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # transform = AddSelfLoop()  # by default, it will first remove self-loops to prevent duplication
    if train_status:
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, default="cora",
                            help="Dataset name ('cora', 'citeseer', 'pubmed').")
        args = parser.parse_args()
        # print(f'Training with DGL built-in GraphConv module.')

        use_my_dataset = True
        # load and preprocess dataset

        # if not use_my_dataset and args.dataset == 'cora':
        #     data = CoraGraphDataset(transform=transform)
        # elif not use_my_dataset and args.dataset == 'citeseer':
        #     data = CiteseerGraphDataset(transform=transform)
        # elif not use_my_dataset and args.dataset == 'pubmed':
        #     data = PubmedGraphDataset(transform=transform)
        # else:
        #     raise ValueError('Unknown dataset: {}'.format(args.dataset))

        # if not use_my_dataset:
        #     g = data[0]
        #     # g = make_graph(1, 7, transform=transform, device=device)
        #     g = g.int().to(device)
        #     features = g.ndata['feat']
        #     labels = g.ndata['label']
        #     masks = g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask']

        # make features and labels, features must be the same

        # adjs = map(lambda x: nx.adjacency_matrix(x), g) # this can make a list of nx.adjacency_matrix
        # feats = [scipy.sparse.identity(adjs[num_time_steps - 1].shape[0]).tocsr()[range(0, x.shape[0]), :] for x in adjs if
        #     x.shape[0] <= adjs[num_time_steps - 1].shape[0]]
        # Identity matrix above, can't be used

        # normalization
        # degs = g.in_degrees().float()
        # norm = torch.pow(degs, -0.5).to(device)
        # norm[torch.isinf(norm)] = 0
        # g.ndata['norm'] = norm.unsqueeze(1)
        deepwalk_T = 1

        # last_Test = 1
        GraphSAGE_T = 0
        if deepwalk_T:

            # DeepWalk_train(train_set)
            files = 0
            if files:

                data = np.load("node_embeddings.npy", allow_pickle=True)
            else:
                node_embeddings = []
                for i in range(6):
                    resulst = make_graph(i + 1, i + 7, device=device)
                    graph = resulst[0][0]
                    # Initialize the DeepWalk model
                    deepwalk = DeepWalk(graph, window_size=5, walk_length=10, num_walks=20, embedding_dim=128)

                    # Get node embeddings
                    node_embeddings.append(deepwalk.get_node_embeddings())
                    walks = deepwalk.generate_walks()
                train_set = [
                    make_features(node_embeddings, 1),
                    make_features(node_embeddings, 2),
                    make_features(node_embeddings, 3),
                    make_features(node_embeddings, 4),
                    make_features(node_embeddings, 5)
                ]
                np.save("node_embeddings.npy", node_embeddings)
                with open("node_embeddings.txt", "a+") as f:
                    for i in range(len(node_embeddings)):
                        for j in range(200):
                            f.writelines(str(j) + "\t")
                            # f.writelines(np.save())
                            # print(node_embeddings[i][j])
                data = node_embeddings
            # print(data)
            # make_features(data,1)
            train_set = [
                make_features(data, 1),
                make_features(data, 2),
                make_features(data, 3),
                make_features(data, 4),
                make_features(data, 5)
            ]
            model = Net().to(device)
            DeepWalk_train(model, train_set)
            test_set = make_features(data, 6)
            print('Evaluating~~~')
            mape, r2 = evaluate_D(model, test_set)
            print("Test mape {:.4f}, r2 {:.4f}".format(mape, r2))
        elif GraphSAGE_T:

            # create GCN model
            in_size = 128
            # out_size = out_put_size
            in_feat = 128
            out_size = 1
            # model = DeepWalk(in_feat, 128, out_size).to(device)
            # model = GraphSAGE(in_feat, 128, out_size).to(device)
            # model = GAT(in_feat, 128, 1, out_size).to(device)
            model = GCN(in_size, 16, out_size).to(device)
            # train_set = [make_graph(1, 1, device=device)]
            train_set = [
                make_graph(1, 7, device=device),
                make_graph(2, 8, device=device),
                make_graph(3, 9, device=device),
                make_graph(4, 10, device=device),
                make_graph(5, 11, device=device),

                #
            ]

            # model training
            print('Training...')
            train(model, train_set)
            model.load_state_dict(load_model(0))
            test_set = make_graph(6, 12, device=device)
            print('Testing...')
            mape, r2 = evaluate(model, test_set)
            print("Test mape {:.4f}, r2 {:.4f}".format(mape, r2))
        else:
            with open('./metapath2vec.txt', 'r') as r:
                day_embeddings = r.readlines()
            print(day_embeddings)
            train_set = [
                make_meta_embeddings(day_embeddings, 1),
                make_meta_embeddings(day_embeddings, 2),
                make_meta_embeddings(day_embeddings, 3),
                make_meta_embeddings(day_embeddings, 4),
                make_meta_embeddings(day_embeddings, 5)

            ]
            model = Net().to(device)
            metapath2vec_train(model, train_set)
            test_set = make_meta_embeddings(day_embeddings, 6)
            print('Evaluating~~~')
            mape, r2 = evaluate_M(model, test_set)
            print("Test mape {:.4f}, r2 {:.4f}".format(mape, r2))
    # # create GCN model
    # in_size = 500
    # out_size = out_put_size
    # model = GCN(in_size, 16, out_size).to(device)
    # model.load_state_dict(load_model(0))
    # test_set = make_graph(6, 12, transform=transform, device=device)
    # # test the model
    # print('Testing...')
    # mape, r2 = evaluate(model, test_set)
    # print("Test mape {:.4f}, r2 {:.4f}".format(mape, r2))
