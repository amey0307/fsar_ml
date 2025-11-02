import torch
from src.dataset import NTUSkeletonDataset
from src.model import STGCN
from src.train import federated_train
import os

def get_all_subjects(data_dir):
    subject_ids = set()
    for file in os.listdir(data_dir):
        if file.endswith('.skeleton'):
            sid = file[:4]
            subject_ids.add(sid)
    return sorted(list(subject_ids))

def main():
    data_dir = "data/nturgb+d_skeletons"
    all_subjects = get_all_subjects(data_dir)
    mid = len(all_subjects) // 2
    client1_subjects = all_subjects[:mid]
    client2_subjects = all_subjects[mid:]

    client1_dataset = NTUSkeletonDataset(data_dir, subject_ids=client1_subjects)
    client2_dataset = NTUSkeletonDataset(data_dir, subject_ids=client2_subjects)

    client_datasets = [client1_dataset, client2_dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    global_model = STGCN(num_class=60).to(device)
    trained_model = federated_train(global_model, client_datasets, epochs=10, batch_size=8, device=device, lr=1e-3, weight_decay=1e-5, scheduler_step=10, scheduler_gamma=0.1)
    torch.save(trained_model.state_dict(), "federated_stgcn.pth")
    print("Model trained and saved.")

if __name__ == "__main__":
    main()
