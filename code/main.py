# Main file to test all approaches and models with the various datasets

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


print("Training GCN on SBM Dataset in full batch")
start_time = time.time()
train_acc, test_acc = run_model_with_full_batch(sbm_data, GCNNet, device, num_epochs=20)
end_time = time.time()
print(f"SBM Dataset - Test Accuracy: {test_acc:.4f}, Time: {end_time - start_time:.2f}s\n")

print("Training GraphSAGENet on SBM Dataset in full batch")
start_time = time.time()
train_acc, test_acc = run_model_with_full_batch(sbm_data, GraphSAGENet, device, num_epochs=20)
end_time = time.time()
print(f"SBM Dataset - Test Accuracy: {test_acc:.4f}, Time: {end_time - start_time:.2f}s\n")

print("Training GCN with neighbor sampling on SBM Dataset")
start_time = time.time()
train_acc, test_acc = run_model_with_neighbor_sampling(sbm_data, GCNNet, device, num_epochs=20)
end_time = time.time()
print(f"SBM Dataset - Test Accuracy: {test_acc:.4f}, Time: {end_time - start_time:.2f}s\n")

print("Training GraphSAGENet with neighbor sampling on SBM Dataset")
start_time = time.time()
train_acc, test_acc = run_model_with_neighbor_sampling(sbm_data, GraphSAGENet, device, num_epochs=20)
end_time = time.time()
print(f"SBM Dataset - Test Accuracy: {test_acc:.4f}, Time: {end_time - start_time:.2f}s\n")

print("Training GCN with graph partitioning on SBM Dataset")
start_time = time.time()
train_acc, test_acc = run_model_with_cluster_loader(sbm_data, GCNNet, device, num_epochs=20)
end_time = time.time()
print(f"SBM Dataset - Test Accuracy: {test_acc:.4f}, Time: {end_time - start_time:.2f}s\n")

print("Training GraphSAGENet with graph partitioning on SBM Dataset")
start_time = time.time()
train_acc, test_acc = run_model_with_cluster_loader(sbm_data, GraphSAGENet, device, num_epochs=20)
end_time = time.time()
print(f"SBM Dataset - Test Accuracy: {test_acc:.4f}, Time: {end_time - start_time:.2f}s\n")

num_nodes = 10000
num_classes = 10
p_intra = 0.1
p_inter = 0.01
feature_dim = 35
sbm_data = generate_sbm_data(num_nodes, num_classes, p_intra, p_inter, feature_dim, seed=42)
sbm_data = split_data(sbm_data, num_classes, train_ratio=0.8, test_ratio=0.2, seed=42)

print("Training GCN on SBM Dataset in full batch")
start_time = time.time()
train_acc, test_acc = run_model_with_full_batch(sbm_data, GCNNet, device, num_epochs=20)
end_time = time.time()
print(f"SBM Dataset - Test Accuracy: {test_acc:.4f}, Time: {end_time - start_time:.2f}s\n")

print("Training GraphSAGENet on SBM Dataset in full batch")
start_time = time.time()
train_acc, test_acc = run_model_with_full_batch(sbm_data, GraphSAGENet, device, num_epochs=20)
end_time = time.time()
print(f"SBM Dataset - Test Accuracy: {test_acc:.4f}, Time: {end_time - start_time:.2f}s\n")

print("Training GCN with neighbor sampling on SBM Dataset")
start_time = time.time()
train_acc, test_acc = run_model_with_neighbor_sampling(sbm_data, GCNNet, device, num_epochs=20)
end_time = time.time()
print(f"SBM Dataset - Test Accuracy: {test_acc:.4f}, Time: {end_time - start_time:.2f}s\n")

print("Training GraphSAGENet with neighbor sampling on SBM Dataset")
start_time = time.time()
train_acc, test_acc = run_model_with_neighbor_sampling(sbm_data, GraphSAGENet, device, num_epochs=20)
end_time = time.time()
print(f"SBM Dataset - Test Accuracy: {test_acc:.4f}, Time: {end_time - start_time:.2f}s\n")

print("Training GCN with graph partitioning on SBM Dataset")
start_time = time.time()
train_acc, test_acc = run_model_with_cluster_loader(sbm_data, GCNNet, device, num_epochs=20)
end_time = time.time()
print(f"SBM Dataset - Test Accuracy: {test_acc:.4f}, Time: {end_time - start_time:.2f}s\n")

print("Training GraphSAGENet with graph partitioning on SBM Dataset")
start_time = time.time()
train_acc, test_acc = run_model_with_cluster_loader(sbm_data, GraphSAGENet, device, num_epochs=20)
end_time = time.time()
print(f"SBM Dataset - Test Accuracy: {test_acc:.4f}, Time: {end_time - start_time:.2f}s\n")

# CORA dataset
cora_data = load_cora_dataset()
cora_data = split_data(cora_data, cora_data.y.max().item() + 1, train_ratio=0.8, test_ratio=0.2, seed=42)

print("Training GCN on CORA Dataset in full batch")
start_time = time.time()
train_acc, test_acc = run_model_with_full_batch(cora_data, GCNNet, device, num_epochs=20)
end_time = time.time()
print(f"CORA Dataset - Test Accuracy: {test_acc:.4f}, Time: {end_time - start_time:.2f}s\n")

print("Training GraphSAGENet on CORA Dataset in full batch")
start_time = time.time()
train_acc, test_acc = run_model_with_full_batch(cora_data, GraphSAGENet, device, num_epochs=20)
end_time = time.time()
print(f"CORA Dataset - Test Accuracy: {test_acc:.4f}, Time: {end_time - start_time:.2f}s\n")

print("Training GCN with neighbor sampling on CORA Dataset")
start_time = time.time()
train_acc, test_acc = run_model_with_neighbor_sampling(cora_data, GCNNet, device, num_epochs=20)
end_time = time.time()
print(f"CORA Dataset - Test Accuracy: {test_acc:.4f}, Time: {end_time - start_time:.2f}s\n")

print("Training GraphSAGENet with neighbor sampling on CORA Dataset")
start_time = time.time()
train_acc, test_acc = run_model_with_neighbor_sampling(cora_data, GraphSAGENet, device, num_epochs=20)
end_time = time.time()
print(f"CORA Dataset - Test Accuracy: {test_acc:.4f}, Time: {end_time - start_time:.2f}s\n")

print("Training GCN with graph partitioning on CORA Dataset")
start_time = time.time()
train_acc, test_acc = run_model_with_cluster_loader(cora_data, GCNNet, device, num_epochs=20)
end_time = time.time()
print(f"CORA Dataset - Test Accuracy: {test_acc:.4f}, Time: {end_time - start_time:.2f}s\n")

print("Training GraphSAGENet with graph partitioning on CORA Dataset")
start_time = time.time()
train_acc, test_acc = run_model_with_cluster_loader(cora_data, GraphSAGENet, device, num_epochs=20)
end_time = time.time()
print(f"CORA Dataset - Test Accuracy: {test_acc:.4f}, Time: {end_time - start_time:.2f}s\n")

# Reddit dataset
reddit_data = load_reddit_dataset()
reddit_data = split_data(reddit_data, reddit_data.y.max().item() + 1, train_ratio=0.8, test_ratio=0.2, seed=42)

print("Training GCN on Reddit Dataset in full batch")
start_time = time.time()
train_acc, test_acc = run_model_with_full_batch(reddit_data, GCNNet, device, num_epochs=20)
end_time = time.time()
print(f"Reddit Dataset - Test Accuracy: {test_acc:.4f}, Time: {end_time - start_time:.2f}s\n")

print("Training GraphSAGENet on Reddit Dataset in full batch")
start_time = time.time()
train_acc, test_acc = run_model_with_full_batch(reddit_data, GraphSAGENet, device, num_epochs=20)
end_time = time.time()
print(f"Reddit Dataset - Test Accuracy: {test_acc:.4f}, Time: {end_time - start_time:.2f}s\n")

print("Training GCN with neighbor sampling on Reddit Dataset")
start_time = time.time()
train_acc, test_acc = run_model_with_neighbor_sampling(reddit_data, GCNNet, device, num_epochs=20)
end_time = time.time()
print(f"Reddit Dataset - Test Accuracy: {test_acc:.4f}, Time: {end_time - start_time:.2f}s\n")

print("Training GraphSAGENet with neighbor sampling on Reddit Dataset")
start_time = time.time()
train_acc, test_acc = run_model_with_neighbor_sampling(reddit_data, GraphSAGENet, device, num_epochs=20)
end_time = time.time()
print(f"Reddit Dataset - Test Accuracy: {test_acc:.4f}, Time: {end_time - start_time:.2f}s\n")

print("Training GCN with graph partitioning on Reddit Dataset")
start_time = time.time()
train_acc, test_acc = run_model_with_cluster_loader(reddit_data, GCNNet, device, num_epochs=20)
end_time = time.time()
print(f"Reddit Dataset - Test Accuracy: {test_acc:.4f}, Time: {end_time - start_time:.2f}s\n")

print("Training GraphSAGENet with graph partitioning on Reddit Dataset")
start_time = time.time()
train_acc, test_acc = run_model_with_cluster_loader(reddit_data, GraphSAGENet, device, num_epochs=20)
end_time = time.time()
print(f"Reddit Dataset - Test Accuracy: {test_acc:.4f}, Time: {end_time - start_time:.2f}s\n")