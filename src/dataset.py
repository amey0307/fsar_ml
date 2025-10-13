import torch
from torch.utils.data import Dataset
import os
from .preprocess import read_skeleton_file, normalize_skeleton
import numpy as np


class NTUSkeletonDataset(Dataset):
    def __init__(self, data_dir, subject_ids=None, transform=None):
        self.files = []
        for root, _, filenames in os.walk(data_dir):
            for file in filenames:
                if file.endswith('.skeleton'):
                    subject_id = file[0:4]  # 'S001', 'S002', etc.
                    if (subject_ids is None) or (subject_id in subject_ids):
                        self.files.append(os.path.join(root, file))
        print(f"Loaded {len(self.files)} files for subjects: {subject_ids}")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        skeleton = read_skeleton_file(file_path)
        skeleton = normalize_skeleton(skeleton)
        max_frames = 100
        num_frames = skeleton.shape[0]
        if num_frames < max_frames:
            pad = np.zeros((max_frames - num_frames, skeleton.shape[1], 3))
            skeleton = np.concatenate((skeleton, pad), axis=0)
        else:
            skeleton = skeleton[:max_frames]
        skeleton = torch.tensor(skeleton, dtype=torch.float32).permute(2, 1, 0)
        label = torch.tensor(0, dtype=torch.long)  # Update label extraction as needed
        if self.transform:
            skeleton = self.transform(skeleton)
        return skeleton, label
