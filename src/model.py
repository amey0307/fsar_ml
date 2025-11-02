import torch
import torch.nn as nn

class STGCN(nn.Module):
    def __init__(self, num_class=60):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=1)
        self.fc = nn.Linear(128 * 25 * 100, num_class)
        
    def forward(self, x):
        # Input x: [batch, 3, 25, 100]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
