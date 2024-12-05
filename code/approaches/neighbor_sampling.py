import torch


def train_neighbor_loader(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_examples = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = criterion(out[:batch.batch_size], batch.y[:batch.batch_size])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_nodes
        total_examples += batch.num_nodes
    return total_loss / total_examples


def evaluate_neighbor_loader(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            pred = out[:batch.batch_size].argmax(dim=1)
            correct += (pred == batch.y[:batch.batch_size]).sum().item()
            total += batch.batch_size
    return correct / total
