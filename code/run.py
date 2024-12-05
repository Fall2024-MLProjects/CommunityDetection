import torch
import torch.nn as nn
from approaches.full_batch import train_full_batch, evaluate
from approaches.neighbor_sampling import train_neighbor_loader, evaluate_neighbor_loader
from approaches.graph_partitioning import train_cluster_loader, evaluate_cluster_loader
from torch_geometric.loader import NeighborLoader, ClusterData, ClusterLoader


def run_model_with_full_batch(data, model_cls, device, num_epochs):
    data = data.to(device)
    model = model_cls(data.num_node_features, 64, data.y.max().item() + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, num_epochs+1):
        loss = train_full_batch(model, data, optimizer, criterion, device)
        train_acc, test_acc = evaluate(model, data, device)
        print(f'Epoch {epoch}, Loss: {loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    train_acc, test_acc = evaluate(model, data, device)
    return train_acc, test_acc


def run_model_with_neighbor_sampling(data, model_cls, device, num_epochs):
    data = data.to(device)
    model = model_cls(data.num_node_features, 64, data.y.max().item() + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    test_idx = data.test_mask.nonzero(as_tuple=False).view(-1)

    # Create NeighborLoaders for training and evaluation
    train_loader = NeighborLoader(
        data,
        num_neighbors=[10, 10],
        batch_size=32,
        input_nodes=train_idx,
        shuffle=True,
    )

    # For evaluation, we can limit the neighbors to a manageable number if necessary
    eval_loader = NeighborLoader(
        data,
        num_neighbors=[10, 10],
        batch_size=64,
        input_nodes=test_idx,
        shuffle=False,
    )

    for epoch in range(1, num_epochs + 1):
        loss = train_neighbor_loader(model, train_loader, optimizer, criterion, device)
        train_acc = evaluate_neighbor_loader(model, train_loader, device)
        test_acc = evaluate_neighbor_loader(model, eval_loader, device)
        print(f'Epoch {epoch}, Loss: {loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    return train_acc, test_acc


def run_model_with_cluster_loader(data, model_cls, device, num_epochs):
    data = data.to(device)
    model = model_cls(data.num_node_features, 64, data.y.max().item() + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # Partition the graph into clusters
    cluster_data = ClusterData(data, num_parts=150, recursive=False)
    train_loader = ClusterLoader(cluster_data, batch_size=32, shuffle=True)
    eval_loader = ClusterLoader(cluster_data, batch_size=32, shuffle=False)

    for epoch in range(1, num_epochs + 1):
        loss = train_cluster_loader(model, train_loader, optimizer, criterion, device)
        train_acc = evaluate_cluster_loader(model, eval_loader, device, mask_name='train_mask')
        test_acc = evaluate_cluster_loader(model, eval_loader, device, mask_name='test_mask')
        print(f'Epoch {epoch}, Loss: {loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    return train_acc, test_acc

