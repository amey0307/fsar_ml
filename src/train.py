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

def federated_train(global_model, client_datasets, epochs=20, batch_size=8, device='cpu', lr=1e-3, weight_decay=1e-5, scheduler_step=10, scheduler_gamma=0.1):
    from .federated import FederatedAveraging
    federated_averaging = FederatedAveraging(global_model)
    criterion = torch.nn.CrossEntropyLoss()

    client_models = [copy.deepcopy(global_model).to(device) for _ in client_datasets]
    client_optimizers = []
    client_schedulers = []
    client_loaders = []

    for i, ds in enumerate(client_datasets):
        optimizer = torch.optim.Adam(client_models[i].parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
        client_optimizers.append(optimizer)
        client_schedulers.append(scheduler)
        client_loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True))

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for i, model in enumerate(client_models):
            loss = train_one_epoch(model, client_loaders[i], client_optimizers[i], criterion, device)
            print(f" Client {i+1} loss: {loss:.4f}")
            client_schedulers[i].step()
        # Aggregate client models
        global_model = federated_averaging.aggregate(client_models)
        # Synchronize global model to clients for next round
        for i in range(len(client_models)):
            client_models[i].load_state_dict(global_model.state_dict())

    return global_model
