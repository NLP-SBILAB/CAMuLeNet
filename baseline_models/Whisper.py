import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import pickle
import os
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class EmotionClassifierCNN(nn.Module):
    def __init__(self):
        super(EmotionClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=1)

        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Dropout layers
        self.dropout = nn.Dropout(0.25)
        
        # Calculate the size of the features after the convolution and pooling layers
        # This size is needed for the first fully connected layer
        self.fc1 = nn.Linear(1920, 128)  # The size 128*8*4 should be adjusted based on the actual output size after convolutions
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 16) 
        self.fc4 = nn.Linear(16, 6) # 6 classes
        self.bn1 = nn.BatchNorm2d(16)  # For the output of the first convolutional layer
        self.bn2 = nn.BatchNorm2d(32)  # For the output of the second convolutional layer
        self.bn3 = nn.BatchNorm2d(64)  # For the output of the third convolutional layer
        self.bn4 = nn.BatchNorm2d(128) # For the output of the fourth convolutional layer

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        x = self.dropout(x)
        x = self.pool(self.bn3(F.relu(self.conv3(x))))
        x = self.dropout(x)
        x = self.pool(self.bn4(F.relu(self.conv4(x))))
        x = self.dropout(x)
    
        # Flatten the output for the fully connected layer
        # Make sure to adjust the flattening based on the actual output size
        x = x.view(-1, 1920) #-> MEDIUM
        #x = x.view(-1, 640) #-> BASE
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)  # No need to apply softmax here if using nn.CrossEntropyLoss
    
        return x