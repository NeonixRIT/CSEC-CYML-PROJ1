import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import csv
from time import perf_counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, MinMaxScaler


def batch_iterate(batch_size, X, y):
    perm = mx.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s : s + batch_size]
        yield X[ids], y[ids]


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(77, 64), nn.ELU(), nn.Linear(64, 32), nn.ELU(), nn.Linear(32, 24), nn.ELU(), nn.Linear(24, 16), nn.ELU(), nn.Linear(16, 8), nn.ELU())
        self.decoder = nn.Sequential(nn.Linear(8, 16), nn.ELU(), nn.Linear(16, 24), nn.ELU(), nn.Linear(24, 32), nn.ELU(), nn.Linear(32, 64), nn.ELU(), nn.Linear(64, 77), nn.ELU())
        self.optimizer = optim.Adam(learning_rate=0.001)

    def __call__(self, X):
        x = self.encoder(X)
        x = self.decoder(x)
        return x


def loss_fn(model, X, y):
    return mx.mean(nn.losses.mse_loss(model(X), X))


def eval_fn(model, X, y):
    return mx.mean(model(X) == X)


def train(num_epochs: int, batch_size: int, Xs_train, ys_train):
    model = Autoencoder()
    mx.eval(model.parameters())
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    for e in range(num_epochs):
        start = perf_counter()
        for X, y in batch_iterate(batch_size, Xs_train, ys_train):
            loss, grad = loss_and_grad_fn(model, X, y)
            model.optimizer.update(model, grad)
            mx.eval(model.parameters(), model.optimizer.state)
        end = perf_counter()
        print(f"Epoch {e} - Loss: {loss.item():.7f}, Time: {end - start:.2f}s")


def main():
    data = None
    with open('data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'r') as f:
        data = list(csv.reader(f))[1:]

    print(f'Using Device: {mx.default_device()}')
    print(mx.metal.device_info())

    Xs = np.array(list(map(lambda row: [float(x) if x.lower() != 'nan' and x.lower() != 'infinity' else 0.0 for x in row[7:-1]], data)), dtype=np.float32)
    ys = np.array(list(map(lambda row: [1] if row[-1] == 'DDoS' else [0], data)), dtype=np.float32)

    scaler = Normalizer()
    Xs = scaler.fit_transform(Xs)
    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(Xs)

    Xs_train, Xs_test, ys_train, ys_test = train_test_split(Xs, ys, test_size=0.4, random_state=1)

    Xs_train = mx.array(Xs_train)
    Xs_test = mx.array(Xs_test)
    ys_train = mx.array(ys_train)
    ys_test = mx.array(ys_test)

    train(300, 2560, Xs_train, ys_train)


if __name__ == '__main__':
    main()
