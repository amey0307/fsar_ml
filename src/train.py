import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in tqdm(dataloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item()
            _, predicted = torch.max(pred, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    acc = correct / total
    return total_loss / len(dataloader), acc

def federated_train(global_model, client_datasets, epochs=5, batch_size=8, device='cpu'):
    from .federated import FederatedAveraging
    federated_averaging = FederatedAveraging(global_model)
    criterion = torch.nn.CrossEntropyLoss()

    client_models = [copy.deepcopy(global_model).to(device) for _ in client_datasets]
    client_optimizers = [torch.optim.Adam(model.parameters(), lr=1e-3) for model in client_models]
    client_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in client_datasets]

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        # Local client updates
        for i, model in enumerate(client_models):
            loss = train_one_epoch(model, client_loaders[i], client_optimizers[i], criterion, device)
            print(f" Client {i+1} loss: {loss:.4f}")

        # Aggregate client models into global model
        global_model = federated_averaging.aggregate(client_models)
        # Synchronize global model to clients for next round
        for i in range(len(client_models)):
            client_models[i].load_state_dict(global_model.state_dict())

    return global_model
