from data import generate_sbm_data, load_cora_dataset, load_reddit_dataset, split_data
import time
from run import run_model_with_full_batch, run_model_with_neighbor_sampling, run_model_with_cluster_loader
import torch
from models import GCNNet, GraphSAGENet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_nodes = 1000
num_classes = 4
p_intra = 0.05
p_inter = 0.005
feature_dim = 16
sbm_data = generate_sbm_data(num_nodes, num_classes, p_intra, p_inter, feature_dim, seed=42)
sbm_data = split_data(sbm_data, num_classes, train_ratio=0.8, test_ratio=0.2, seed=42)


print("Training GNN on SBM Dataset in full batch")
start_time = time.time()
train_acc, test_acc = run_model_with_full_batch(sbm_data, GCNNet, device, num_epochs=20)
end_time = time.time()
print(f"SBM Dataset - Test Accuracy: {test_acc:.4f}, Time: {end_time - start_time:.2f}s\n")

print("Training GNN on SBM Dataset with Neighbor Sampling")
start_time = time.time()
train_acc, test_acc = run_model_with_neighbor_sampling(sbm_data, GCNNet, device, num_epochs=20)
end_time = time.time()
print(f"SBM Dataset - Test Accuracy: {test_acc:.4f}, Time: {end_time - start_time:.2f}s\n")

print("Training GNN on SBM Dataset with Graph Partitioning")
start_time = time.time()
train_acc, test_acc = run_model_with_cluster_loader(sbm_data, GCNNet, device, num_epochs=20)
end_time = time.time()
print(f"SBM Dataset - Test Accuracy: {test_acc:.4f}, Time: {end_time - start_time:.2f}s\n")
