import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Reddit


def generate_sbm_data(num_nodes, num_classes, p_intra, p_inter, feature_dim, seed=None):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Assigning nodes to communities
    labels = np.random.randint(0, num_classes, size=num_nodes)

    # Initialize adjacency matrix
    adj_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if labels[i] == labels[j]:
                if np.random.rand() < p_intra:
                    adj_matrix[i, j] = adj_matrix[j, i] = 1
            else:
                if np.random.rand() < p_inter:
                    adj_matrix[i, j] = adj_matrix[j, i] = 1

    # Generating random node features
    features = np.random.randn(num_nodes, feature_dim)

    # Converting to PyTorch Geometric Data object
    edge_index = torch.tensor(np.array(np.nonzero(adj_matrix)), dtype=torch.long)
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data


def split_data(data, num_classes, train_ratio=0.8, test_ratio=0.2, seed=None):
    if seed is not None:
        np.random.seed(seed)

    num_nodes = data.num_nodes
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)
    train_size = int(train_ratio * num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[indices[:train_size]] = True
    test_mask[indices[train_size:]] = True
    data.train_mask = train_mask
    data.test_mask = test_mask
    return data


def load_cora_dataset():
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    return data


def load_reddit_dataset():
    dataset = Reddit(root='/tmp/Reddit')
    data = dataset[0]
    return data

