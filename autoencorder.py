"""
Using reference code found here:
https://b-nova.com/en/home/content/anomaly-detection-with-random-forest-and-pytorch/
"""

import csv
import json
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim

# from itertools import batched
from pprint import pprint
from time import perf_counter

from torch.nn.functional import mse_loss
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, MinMaxScaler


# Prevent warnings from being printed
warnings.filterwarnings('ignore')


def batch_iterate(batch_size, X: torch.Tensor, y: torch.Tensor, device):
    perm = torch.from_numpy(np.random.permutation(y.size(0))).to(device)
    for s in range(0, y.size(0), batch_size):
        ids = perm[s:s + batch_size]
        yield X[ids], y[ids]


class NetFlowDataSet(Dataset):
    """
    Load the dataset from a CSV file and preprocess it
    Treats infinities and NaNs as 0
    Sets BENIGN to 0 and DDoS to 1
    Normalizes and scales the data

    Excludes data like IP addresses, ports, and timestamps to help with generalization
    """

    def __init__(self, csv_file, device):
        self.device = device
        with open(csv_file, 'r') as f:
            temp = list(csv.reader(f))
            self.columns = [col_tag.strip() for col_tag in temp[0][7:-1]]
            temp = temp[1:]
            self.Xs = np.array(list(map(lambda row: [float(x) if x.lower() != 'nan' and x.lower() != 'infinity' else 0.0 for x in row[7:-1]], temp)), dtype=np.float32)
            self.ys = np.array(list(map(lambda row: [1] if row[-1] == 'DDoS' else [0], temp)), dtype=np.float32)

            scaler = Normalizer()
            self.Xs = scaler.fit_transform(self.Xs)
            scaler = MinMaxScaler()
            self.Xs = scaler.fit_transform(self.Xs)

            self.Xs = torch.from_numpy(self.Xs).to(device)
            self.ys = torch.from_numpy(self.ys).to(device)


    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, idx):
        return self.Xs[idx], self.ys[idx]


# Define the autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self, device):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(77, 64), nn.ELU(True), nn.Linear(64, 32), nn.ELU(True), nn.Linear(32, 24), nn.ELU(True), nn.Linear(24, 16), nn.ELU(True), nn.Linear(16, 8), nn.ELU(True), nn.LayerNorm(8))
        self.decoder = nn.Sequential(nn.Linear(8, 16), nn.ELU(True), nn.Linear(16, 24), nn.ELU(True), nn.Linear(24, 32), nn.ELU(True), nn.Linear(32, 64), nn.ELU(True), nn.Linear(64, 77), nn.ELU(True))
        self.device = device
        self.loss_fn = nn.MSELoss().to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def reset_parameters(self):
        for layer in self.encoder:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.decoder:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def save(self, file_name='autoencoder.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='autoencoder.pth'):
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)
        self.load_state_dict(torch.load(file_name))
        self.to(self.device)


def train_autoencoder(autoencoder: Autoencoder, batch_size: int, Xs_train: torch.Tensor, ys_train: torch.Tensor, desired_loss=0.000014):
    autoencoder.train()
    autoencoder.to(autoencoder.device)

    print('\nTraining the autoencoder...')
    epoch = 0
    desired_loss = torch.tensor([desired_loss], device=autoencoder.device).item()
    loss = torch.tensor([100], device=autoencoder.device)
    previous_loss = torch.tensor([100], device=autoencoder.device)
    while loss.item() > desired_loss:
        start = perf_counter()
        for X, _ in batch_iterate(batch_size, Xs_train, ys_train, device=autoencoder.device):
            autoencoder.optimizer.zero_grad()
            preds: torch.Tensor = autoencoder(X)
            loss: torch.Tensor = autoencoder.loss_fn(preds, X)
            loss.backward()
            autoencoder.optimizer.step()
        end = perf_counter()
        if (loss.item() != 100 and previous_loss.item() != 100) and (loss.item() / previous_loss.item() > 3):
            print('Gradiant explosion detected restarting training...')
            os.execl(sys.executable, sys.executable, *sys.argv)
        previous_loss = loss
        print(f'Epoch [{epoch+1}/???], Loss: {loss.item():.7f}, Time: {end - start:.2f}s')
        epoch += 1
    # autoencoder.save(f'autoencoder_{epoch}.pth')


def calculate_reconstruction_loss(data: torch.Tensor, model: nn.Module):
    with torch.no_grad():
        reconstructed = model(data)
        # Compute reconstruction loss (e.g., MSE))
        loss = mse_loss(reconstructed, data, reduction='none').mean(dim=1)
    return loss


def perturb_feature(data: torch.Tensor, feature_idx: int, perturbation_type: str = 'zero'):
    perturbed_data = data.clone()  # Create a copy of the data
    if perturbation_type == 'zero':
        perturbed_data[:, feature_idx] = 0  # Set the feature's value to zero in feature_idx column for every row
    elif perturbation_type == 'noise':
        perturbed_data[:, feature_idx] += torch.randn_like(perturbed_data[:, feature_idx]).item()  # Add noise
    return perturbed_data


def find_important_features(autoencoder: Autoencoder, full_dataset: NetFlowDataSet):
    """
    Find the most important features according to the autoencoder using perturbation

    According the this model, the 8 most important features seem to be:
    1:   Flow Bytes/s (0.00339001234715397)
    2:   Idle Max (0.002881238275222131)
    3:   Flow IAT Max (0.0028654686557274545)
    4:   Idle Mean (0.0028314162427705014)
    5:   Fwd IAT Max (0.0026891995603364194)
    6:   Fwd IAT Total (0.0024685119660716737)
    7:   Flow Duration (0.0022174993446242297)
    8:   Idle Min (0.0019206720553484047)
    9:   Packet Length Variance (0.001180395172923454)
    10:  Bwd IAT Max (0.000999427555143484)
    11:  Fwd IAT Std (0.0009712725095596397)
    12:  Flow IAT Std (0.0008513370212313021)
    13:  Bwd IAT Total (0.0007719564400758827)
    14:  Flow IAT Mean (0.0006468119481723988)
    15:  Flow Packets/s (0.0005138178039487684)
    16:  Fwd IAT Mean (0.00040708279448153917)
    17:  Bwd IAT Std (0.0003659140584204579)
    18:  Flow IAT Min (0.00034386048537271563)
    19:  Idle Std (0.00029638541491294745)
    20:  Bwd IAT Mean (0.00012512591820268426)
    21:  Fwd IAT Min (7.250333328556735e-05)
    22:  Fwd Packets/s (4.64789573015878e-05)
    23:  Bwd Packet Length Std (3.160765299980994e-05)
    24:  Bwd IAT Min (2.4045015379670076e-05)
    25:  Bwd Packet Length Max (1.668739969318267e-05)
    26:  Init_Win_bytes_forward (1.4613904568250291e-05)
    27:  Bwd Packets/s (1.419953150616493e-05)
    28:  Bwd Packet Length Min (1.0935187674476765e-05)
    29:  Active Max (9.469118595006876e-06)
    30:  Active Mean (7.987788194441237e-06)
    31:  Active Min (7.674498192500323e-06)
    32:  Bwd Packet Length Mean (6.861964720883407e-06)
    33:  Avg Bwd Segment Size (6.8382541940081865e-06)
    34:  PSH Flag Count (5.146632247488014e-06)
    35:  Init_Win_bytes_backward (4.318275387049653e-06)
    36:  Max Packet Length (9.122686606133357e-07)
    37:  Packet Length Std (7.146627467591316e-07)
    38:  Subflow Bwd Bytes (6.848731572972611e-07)
    39:  Total Length of Bwd Packets (6.725949788233265e-07)
    40:  Average Packet Size (5.329566192813218e-07)
    41:  Packet Length Mean (5.271667760098353e-07)
    42:  Total Length of Fwd Packets (4.918038030155003e-07)
    43:  Fwd PSH Flags (4.901921784039587e-07)
    44:  Subflow Fwd Bytes (4.891280696028844e-07)
    45:  SYN Flag Count (4.7578032535966486e-07)
    46:  Down/Up Ratio (4.39451468992047e-07)
    47:  ACK Flag Count (4.314897523727268e-07)
    48:  Fwd Packet Length Mean (4.2510873754508793e-07)
    49:  Avg Fwd Segment Size (4.2220563045702875e-07)
    50:  Min Packet Length (3.596796886995435e-07)
    51:  Fwd Packet Length Max (3.5495031625032425e-07)
    52:  Fwd Header Length (3.4491131373215467e-07)
    53:  Fwd Header Length (3.4037657314911485e-07)
    54:  min_seg_size_forward (3.3094329410232604e-07)
    55:  Fwd Packet Length Min (3.2698881113901734e-07)
    56:  Total Fwd Packets (2.883007255150005e-07)
    57:  Subflow Fwd Packets (2.869510353775695e-07)
    58:  Total Backward Packets (2.7831811166834086e-07)
    59:  Subflow Bwd Packets (2.747856342466548e-07)
    60:  act_data_pkt_fwd (2.617816790007055e-07)
    61:  Active Std (2.3100255930330604e-07)
    62:  FIN Flag Count (1.9139952200930566e-07)
    63:  Fwd Packet Length Std (1.8021819414570928e-07)
    64:  Bwd Header Length (1.303615135839209e-07)
    65:  Bwd PSH Flags (-1.8189894035458565e-12)
    66:  Fwd URG Flags (-1.8189894035458565e-12)
    67:  Bwd URG Flags (-1.8189894035458565e-12)
    68:  CWE Flag Count (-1.8189894035458565e-12)
    69:  Fwd Avg Bytes/Bulk (-1.8189894035458565e-12)
    70:  Fwd Avg Packets/Bulk (-1.8189894035458565e-12)
    71:  Fwd Avg Bulk Rate (-1.8189894035458565e-12)
    72:  Bwd Avg Bytes/Bulk (-1.8189894035458565e-12)
    73:  Bwd Avg Packets/Bulk (-1.8189894035458565e-12)
    74:  Bwd Avg Bulk Rate (-1.8189894035458565e-12)
    75:  URG Flag Count (-2.3557731765322387e-08)
    76:  RST Flag Count (-3.5743232729146257e-07)
    77:  ECE Flag Count (-3.5899211070500314e-07)
    """
    # Find which features are the most important using Perturbation
    print('\nCalculating feature importance using perturbation...')
    autoencoder.eval()

    full_dataloader = DataLoader(full_dataset, batch_size=2560, shuffle=False)

    # Collect baseline loss for the entire dataset
    baseline_losses = []
    for X, _ in full_dataloader:
        baseline_losses.append(calculate_reconstruction_loss(X, autoencoder))

    baseline_loss = torch.cat(baseline_losses).mean().item()  # Baseline reconstruction loss

    num_features = len(full_dataset.columns)
    feature_importance = dict()  # Store importance scores for each feature

    for feature_idx in range(num_features):  # Loop over each feature
        print(f'Calculating importance of feature {full_dataset.columns[feature_idx]} ({feature_idx + 1}/{len(full_dataset.columns)})...')
        perturbed_losses = []
        for X, _ in full_dataset:
            if X.dim() == 1:
                X = torch.from_numpy(np.array([X]))
            perturbed_data = perturb_feature(X, feature_idx, perturbation_type='zero')
            perturbed_losses.append(calculate_reconstruction_loss(perturbed_data, autoencoder))

        perturbed_loss = torch.cat(perturbed_losses).mean().item()

        # Calculate the importance of the feature as the difference in reconstruction loss
        feature_importance[feature_idx] = (full_dataset.columns[feature_idx], perturbed_loss - baseline_loss)

    # Visualize the feature importance
    plt.figure(figsize=(16, 8))
    plt.bar([val[0] for val in feature_importance.values()], [val[1] for val in feature_importance.values()])
    plt.xticks(rotation=90)
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.show()

    # Print feature importance dict in descending order of importance
    top_features = {key: val for key, val in sorted(feature_importance.items(), key=lambda x: x[1][1], reverse=True)}
    print('Most important features:')
    pprint(top_features, sort_dicts=False)

    # Save json dict of label to feature importance
    with open('feature_idx_to_importance_and_label.json', 'w') as f:
        json.dump(top_features, f)


class Classifier(nn.Module):
    def __init__(self, device):
        super(Classifier, self).__init__()
        self.layers = nn.Sequential(nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 5), nn.ReLU(), nn.Linear(5, 4), nn.ReLU(), nn.Linear(4, 2), nn.ReLU(), nn.Linear(2, 1), nn.Sigmoid())
        self.to(device)
        self.device = device
        self.loss_fn = nn.BCELoss().to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x):
        return self.layers(x)

    def save(self, file_name='classifier.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='classifier.pth'):
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)
        self.load_state_dict(torch.load(file_name))
        self.to(self.device)


def is_ddos(pred: torch.Tensor, threshold: float = 0.5):
    """
    Returns 1 if the prediction is greater than or equal to the threshold, else 0

    threshold should be between 0 and 1
    threshold = 0.5 is equivalent to rounding the prediction to the nearest integer
    """
    return int(pred.item() >= threshold)


def train_classifier(autoencoder: Autoencoder, classifier: Classifier, batch_size: int, Xs_train: torch.Tensor, ys_train: torch.Tensor, Xs_test: torch.Tensor, ys_test: torch.Tensor, desired_accuracy=99.7):
    autoencoder.eval()
    classifier.train()
    classifier.to(autoencoder.device)

    print('\nTraining the classifier...')
    epoch = 0
    accuracy = 0
    while accuracy < desired_accuracy:
        start = perf_counter()
        for X, y in batch_iterate(batch_size, Xs_train, ys_train, device=autoencoder.device):
            classifier.optimizer.zero_grad()
            features = autoencoder.encoder(X)
            preds: torch.Tensor = classifier(features)
            loss: torch.Tensor = classifier.loss_fn(preds, y)
            loss.backward()
            classifier.optimizer.step()
        classifier.eval()
        accuracy = 0
        with torch.no_grad():
            accuracy = classification_report(ys_test.numpy(), np.array([is_ddos(x) for x in classifier(autoencoder.encoder(Xs_test)).numpy()]), output_dict=True)['accuracy'] * 100
        epoch += 1
        end = perf_counter()
        classifier.train()
        print(f'Epoch [{epoch+1}/???], Loss: {loss.item():.7f}, Accuracy: {accuracy:.2f}%, Time: {end - start:.2f}s')
    # classifier.save(f'classifier_{epoch}.pth')


def test_classifier(autoencoder: Autoencoder, classifier: Classifier, dataset: Dataset):
    autoencoder.eval()
    classifier.eval()

    print('\nEvaluating the classifier...')
    y_pred = []
    y_test = []
    for i in range(len(dataset)):
        X, y = dataset[i]
        features = autoencoder.encoder(X)
        pred = classifier(features)
        y_pred.append(is_ddos(pred, threshold=0.5))
        y_test.append(y.item())

    print(classification_report(y_test, y_pred))
    print()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion matrix')
    plt.show()


def main():
    # Get device
    # CPU is faster for me than MPS, so I'm using CPU, uncomment the line below to use GPU and comment the line below it
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    # device = torch.device('cpu')
    print(f'Using Device: {device}')

    # Load the data
    print('Loading data...', end='')
    csv_file_path = 'data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
    full_dataset = NetFlowDataSet(csv_file_path, device=device)
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(full_dataset.Xs, full_dataset.ys, test_size=0.4, shuffle=True)
    batch_size = 2560
    print('done.')

    # Initialize the models
    autoencoder = Autoencoder(device)
    classifier = Classifier(device)

    # Train the autoencoder
    train_autoencoder(autoencoder, batch_size, Xs_train, ys_train)

    # Train the classifier
    train_classifier(autoencoder, classifier, batch_size, Xs_train, ys_train, Xs_test, ys_test)

    # Test the models
    # autoencoder.load('autoencoder_405.pth')
    # classifier.load('classifier_1323.pth')
    test_classifier(autoencoder, classifier, full_dataset)

    # Find the most important features
    # find_important_features(autoencoder, full_dataset)


if __name__ == '__main__':
    main()
