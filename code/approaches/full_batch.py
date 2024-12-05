import torch


def train_full_batch(model, data, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, data, device):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct_train = (pred[data.train_mask] == data.y[data.train_mask]).sum().item()
        correct_test = (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
        train_acc = correct_train / data.train_mask.sum().item()
        test_acc = correct_test / data.test_mask.sum().item()
    return train_acc, test_acc
