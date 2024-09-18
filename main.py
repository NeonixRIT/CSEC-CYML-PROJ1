import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
import torch.utils
import torch.utils.data

from model import NeuralNetwork, NNTrainer
from torch.utils.data import DataLoader, Dataset
from pandas import read_csv
from time import perf_counter

import warnings
warnings.filterwarnings('ignore')

class PacketFlowDataset(Dataset):
    def __init__(self, csv_file):
        self.data = read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        get the idx-th data

        Best parameters:
            Flow Duration
            Bwd Packet Length Std
            Flow IAT Std
            Avg Packet Size

        Label:
            0: Benign
            1: DDoS
        """
        row = self.data.loc[idx]
        # values = np.array([row[7], row[19], row[23], row[55]], dtype=np.float32) # 4 features
        values = np.array(row[7:-1], dtype=np.float32) # 77 features
        is_ddos = int(row[-1] == 'DDOS')
        is_benign = int(not is_ddos)
        X = torch.tensor(values, dtype=torch.float32)
        y = torch.tensor([is_benign, is_ddos], dtype=torch.float32)
        return X, y


def main():
    # Load data
    data_path = 'data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
    dataset = PacketFlowDataset(data_path)

    # Split data into train and test
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.6), int(len(dataset) * 0.4)])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=True)

    # Create Model

    # Model using 77 features as input
    # Structure is arbitrary (trial and error)
    layers = [
        nn.LayerNorm(77),
        nn.Linear(77, 100),
        nn.ReLU(),
        nn.LayerNorm(100),
        nn.Linear(100, 80),
        nn.ReLU(),
        nn.Linear(80, 20),
        nn.ReLU(),
        nn.Linear(20, 4),
        nn.ReLU(),
        nn.Linear(4, 2)
    ]

    # Model using only 4 features supposed to be the best parameters based
    # on the papaer
    # Structure is arbitrary (trial and error)
    # layers = [
    #     nn.LayerNorm(4),
    #     nn.Linear(4, 80),
    #     nn.ReLU(),
    #     nn.LayerNorm(80),
    #     nn.Linear(80, 10),
    #     nn.ReLU(),
    #     nn.Linear(10, 2),
    # ]
    model = NeuralNetwork(layers)
    model.train()
    trainer = NNTrainer(model, learning_rate=1e-3)

    # Train model
    print()
    epochs = 5
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}\n-------------------------------')
        batch_time = 0
        batch_size = 1000
        for batch, (X, y) in enumerate(train_dataloader):
            start = perf_counter()
            loss = trainer.train_step(X, y)
            end = perf_counter()
            batch_time += end - start
            if batch % batch_size == 0 or batch >= len(train_dataloader) - 1:
                avg_step_time = batch_time / batch_size
                current = batch * len(X)
                print(f'loss: {loss:>7f}  [{current:>5d}/{len(train_dataloader.dataset)}] Time: {avg_step_time * 1000:.4f}ms/step')
                batch_time = 0

    print('Done!')
    # Save model
    model.save()

    # Test model
    # model.load('model/model.pth', train=False)
    model.eval()

    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss, correct = 0, 0
    batch_size = 1000
    with torch.no_grad():
        for batch, (X, y) in enumerate(test_dataloader):
            X: torch.Tensor = X.to(trainer.device)
            y: torch.Tensor = y.to(trainer.device)
            pred: torch.Tensor = model(X)
            test_loss += trainer.loss_fn(pred, y).item()
            correct += int(pred.argmax().item() == y.argmax().item())
            if batch % batch_size == 0 or batch >= len(test_dataloader) - 1:
                current = batch * len(X)
                print(f'[{current:>5d}/{size}]\r')
    test_loss /= num_batches
    correct /= size
    print(f'Test Error: \n Accuracy: {(batch_size * correct):>0.1f}%, Avg loss: {test_loss:>8f}\n')


if __name__ == '__main__':
    main()
