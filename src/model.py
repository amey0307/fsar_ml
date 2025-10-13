import torch
import torch.nn as nn
import torch.nn.functional as F

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_joints=25):
        super(STGCNBlock, self).__init__()
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9,1), padding=(4,0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        y = self.gcn(x)
        y = self.tcn(y)
        if self.downsample:
            x = self.downsample(x)
        return F.relu(x + y)

class STGCN(nn.Module):
    def __init__(self, in_channels=3, num_class=60, num_joints=25):
        super(STGCN, self).__init__()
        self.layer1 = STGCNBlock(in_channels, 64, num_joints)
        self.layer2 = STGCNBlock(64, 128, num_joints)
        self.layer3 = STGCNBlock(128, 256, num_joints)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, num_class)

    def forward(self, x):
        # x shape: (batch, channels=3, joints=25, frames=100)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out
