"""
Sourced from: Modified from https://github.com/patrickloeber/snake-ai-pytorch/
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
import random
from collections import deque


class NeuralNetwork(nn.Module):
    def __init__(self, layers: list):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, model_path, train=False):
        self.load_state_dict(torch.load(model_path))
        if not train:
            self.eval()


class NNTrainer:
    def __init__(self, model: NeuralNetwork, optimizer=optim.SGD, loss_fn=nn.CrossEntropyLoss, learning_rate=1e-3, device=None):
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
            print(f'Using {self.device} device')
        self.model.to(self.device)
        self.optimizer = optimizer(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = loss_fn()

    def train_step(self, X: torch.Tensor, y: torch.Tensor):
        X, y = X.to(self.device), y.to(self.device)

        # Get prediction and calculate loss
        pred = self.model(X)
        loss = self.loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    # def train(self, dataloader):
    #     size = len(dataloader.dataset)
    #     for batch, (x, y) in enumerate(dataloader):
    #         print(batch, x.shape, y.shape)
