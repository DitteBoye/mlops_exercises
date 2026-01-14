import torch
import torch.nn as nn
import torch.nn.functional as F


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)

        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Expected input to be a 4D tensor")

        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError("Expected input shape [batch_size, 1, 28, 28]")
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.mean(dim=[2, 3])
        x = self.dropout(x)
        return self.fc1(x)