import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
class Resnet(nn.Module):
    def __init__(self, num_class, pretrained=False):
        super().__init__()
        self.model = models.resnet18(pretrained=pretrained)
        self.model.fc  = nn.Linear(512, num_class, bias=True)

    def forward(self, x):
        return self.model(x)


class ResnetPretrained(nn.Module):
    
    def __init__(self, num_class, add_new_layer):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.fc = None
        
        if not add_new_layer:
            self.model.fc  = nn.Linear(512, num_class)
        else:
            self.fc = nn.Linear(1000,num_class)
    
    def forward(self, x):
        x  = self.model(x)
        if self.fc is not None:
            return self.fc(x)
    