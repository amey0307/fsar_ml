import torch
from src.dataset import NTUSkeletonDataset
from src.model import STGCN
from src.train import federated_train

def main():
    # Path to folder containing all .skeleton files
    data_dir = "data/nturgb+d_skeletons"
    
    # Define subjects for splitting clients (S001-S017 for client 1, S018-S032 for client 2)
    client1_subjects = ['S{:03d}'.format(i) for i in range(1, 18)]    # S001 to S017
    client2_subjects = ['S{:03d}'.format(i) for i in range(18, 33)]   # S018 to S032

    # Create federated client datasets
    client1_dataset = NTUSkeletonDataset(data_dir, subject_ids=client1_subjects)
    client2_dataset = NTUSkeletonDataset(data_dir, subject_ids=client2_subjects)

    # Optionally print dataset sizes for debugging
    print(f"Client 1 dataset size: {len(client1_dataset)}")
    print(f"Client 2 dataset size: {len(client2_dataset)}")

    client_datasets = [client1_dataset, client2_dataset]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the skeleton recognition model
    global_model = STGCN(num_class=60).to(device)  # Adjust num_class if needed for the dataset

    # Train with federated averaging
    trained_model = federated_train(global_model, client_datasets, epochs=3, batch_size=4, device=device)

    # Save model
    torch.save(trained_model.state_dict(), "federated_stgcn.pth")
    print("Model trained and saved.")

if __name__ == "__main__":
    main()
