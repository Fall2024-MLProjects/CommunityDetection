import torch


def train_cluster_loader(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_examples = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        # Compute loss only on nodes in the training mask within the batch
        train_mask = batch.train_mask
        if train_mask.sum() == 0:
            continue  # Skip batches with no training nodes
        loss = criterion(out[train_mask], batch.y[train_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * train_mask.sum().item()
        total_examples += train_mask.sum().item()
    return total_loss / total_examples


def evaluate_cluster_loader(model, loader, device, mask_name='test_mask'):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            # Get the appropriate mask (train_mask, val_mask, test_mask)
            batch_mask = getattr(batch, mask_name)
            if batch_mask.sum() == 0:
                continue  # Skip batches with no nodes in the mask
            pred = out[batch_mask].argmax(dim=1)
            correct += (pred == batch.y[batch_mask]).sum().item()
            total += batch_mask.sum().item()
    return correct / total
