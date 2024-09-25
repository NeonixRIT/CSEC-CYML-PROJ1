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
        self.sequantial_stack = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequantial_stack(x)

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
    def __init__(self, model: NeuralNetwork, optimizer=optim.Adam, loss_fn=nn.MSELoss, learning_rate=1e-3):
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer = optimizer(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = loss_fn()

    def train_step(self, X: torch.Tensor, y: torch.Tensor):
        # Get prediction and calculate loss
        pred = self.model(X)
        loss = self.loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    def train(self, training_dataloader):
        pass

    def test(self, testing_dataloader):
        pass
