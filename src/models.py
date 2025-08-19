#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out



class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        # First block
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        # Second block  
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        # Third block
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.25)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, args.num_classes)

    def forward(self, x):
        # First block
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second block
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third block
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Flatten and fully connected
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

class CNNCifar100(nn.Module):
    def __init__(self, args):
        super(CNNCifar100, self).__init__()
        # Enhanced architecture for CIFAR-100's 100 classes
        
        # First block - larger filters for more feature extraction
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)  # Increased from 32
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        
        # Second block
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  # Increased from 64
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)
        
        # Third block - Additional for CIFAR-100 complexity
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.25)
        
        # Fully connected layers - larger for 100 classes
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)  # Increased capacity
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout5 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, args.num_classes)  # Should be 100 for CIFAR-100

    def forward(self, x):
        # First block
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second block
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third block
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Flatten and fully connected
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = F.relu(self.fc2(x))
        x = self.dropout5(x)
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1) 

# class CNNCifar(nn.Module):
#     def __init__(self, args):
#         super(CNNCifar, self).__init__()
#         # Bigger architecture for CIFAR-100
#         self.conv1 = nn.Conv2d(3, 64, 3, padding=1)    # More filters
#         self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.dropout1 = nn.Dropout2d(0.1)              # Less dropout
        
#         self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
#         self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.dropout2 = nn.Dropout2d(0.1)
        
#         self.conv5 = nn.Conv2d(128, 256, 3, padding=1)  # Even more filters
#         self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
#         self.pool3 = nn.MaxPool2d(2, 2)
#         self.dropout3 = nn.Dropout2d(0.1)
        
#         # BIGGER FC layers for 100 classes
#         self.fc1 = nn.Linear(256 * 4 * 4, 1024)        # Bigger FC
#         self.dropout4 = nn.Dropout(0.3)
#         self.fc2 = nn.Linear(1024, args.num_classes)

#     def forward(self, x):
#         # First block
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = self.pool1(x)
#         x = self.dropout1(x)
        
#         # Second block
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         x = self.pool2(x)
#         x = self.dropout2(x)
        
#         # Third block
#         x = F.relu(self.conv5(x))
#         x = F.relu(self.conv6(x))
#         x = self.pool3(x)
#         x = self.dropout3(x)
        
#         # Flatten and fully connected
#         x = x.view(-1, 256 * 4 * 4)
#         x = F.relu(self.fc1(x))
#         x = self.dropout4(x)
#         x = self.fc2(x)
        
#         return F.log_softmax(x, dim=1)
    
# class CNNCifar(nn.Module):
#     def __init__(self, args):
#         super(CNNCifar, self).__init__()
#         # Adam-optimized architecture
#         self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
#         self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.dropout1 = nn.Dropout2d(0.1)  # MUCH LOWER
        
#         self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.dropout2 = nn.Dropout2d(0.1)  # MUCH LOWER
        
#         # Simplified FC layers
#         self.fc1 = nn.Linear(256 * 8 * 8, 512)
#         self.dropout3 = nn.Dropout(0.2)  # MUCH LOWER (was 0.5)
#         self.fc2 = nn.Linear(512, args.num_classes)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = self.pool1(x)
#         x = self.dropout1(x)
        
#         x = F.relu(self.conv3(x))
#         x = self.pool2(x)
#         x = self.dropout2(x)
        
#         x = x.view(-1, 256 * 8 * 8)
#         x = F.relu(self.fc1(x))
#         x = self.dropout3(x)
#         x = self.fc2(x)
        
#         return F.log_softmax(x, dim=1)

# class CNNCifar(nn.Module):
#     def __init__(self, args):
#         super(CNNCifar, self).__init__()
#         # First block
#         self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
#         self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.dropout1 = nn.Dropout2d(0.25)
        
#         # Second block  
#         self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
#         self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.dropout2 = nn.Dropout2d(0.25)
        
#         # Third block
#         self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
#         self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
#         self.pool3 = nn.MaxPool2d(2, 2)
#         self.dropout3 = nn.Dropout2d(0.25)
        
#         # ✅ DYNAMIC CALCULATION of flattened size
#         self.feature_size = self._get_conv_output_size()
        
#         # Fully connected layers
#         self.fc1 = nn.Linear(self.feature_size, 512)
#         self.dropout4 = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(512, args.num_classes)

#     def _get_conv_output_size(self):
#         """Calculate the output size after convolutions"""
#         with torch.no_grad():
#             dummy_input = torch.zeros(1, 3, 32, 32)  # CIFAR input size
#             x = self._forward_conv_only(dummy_input)
#             return x.numel()
    
#     def _forward_conv_only(self, x):
#         """Forward pass through conv layers only"""
#         # First block
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = self.pool1(x)
#         x = self.dropout1(x)
        
#         # Second block
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         x = self.pool2(x)
#         x = self.dropout2(x)
        
#         # Third block
#         x = F.relu(self.conv5(x))
#         x = F.relu(self.conv6(x))
#         x = self.pool3(x)
#         x = self.dropout3(x)
        
#         return x.view(x.size(0), -1)

#     def forward(self, x):
#         # Conv layers
#         x = self._forward_conv_only(x)
        
#         # Fully connected
#         x = F.relu(self.fc1(x))
#         x = self.dropout4(x)
#         x = self.fc2(x)
        
#         return F.log_softmax(x, dim=1)