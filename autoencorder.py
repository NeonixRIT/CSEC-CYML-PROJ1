"""
Using reference code found here:
https://b-nova.com/en/home/content/anomaly-detection-with-random-forest-and-pytorch/
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import csv

from pprint import pprint
from torch.nn.functional import mse_loss
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import Normalizer, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

LABELS = ['BENIGN', 'DDoS']


class NetFlowDataSet(Dataset):
    """
    Load the dataset from a CSV file and preprocess it
    Treats infinities and NaNs as 0
    Sets BENIGN to 0 and DDoS to 1
    Normalizes and scales the data

    Excludes data like IP addresses, ports, and timestamps to help with generalization
    """

    def __init__(self, csv_file):
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

    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, idx):
        return torch.tensor(self.Xs[idx], dtype=torch.float32), torch.tensor(self.ys[idx], dtype=torch.float32)


# Define the autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(77, 64), nn.ELU(True), nn.Linear(64, 32), nn.ELU(True), nn.Linear(32, 24), nn.ELU(True), nn.Linear(24, 16), nn.ELU(True), nn.Linear(16, 8), nn.ELU(True))
        self.decoder = nn.Sequential(nn.Linear(8, 16), nn.ELU(True), nn.Linear(16, 24), nn.ELU(True), nn.Linear(24, 32), nn.ELU(True), nn.Linear(32, 64), nn.ELU(True), nn.Linear(64, 77), nn.ELU(True))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

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


def train_autoencoder(autoencoder: Autoencoder, train_dataset: DataLoader, desired_loss=0.000014):
    autoencoder.train()

    print('\nTraining the autoencoder...')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)
    loss = torch.tensor(1.0)
    epoch = 0
    while loss.item() > desired_loss:
        for _, (X, _) in enumerate(train_dataset):
            optimizer.zero_grad()
            preds: torch.Tensor = autoencoder(X)
            loss: torch.Tensor = criterion(preds, X)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/???], Loss: {loss.item():.7f}')
        epoch += 1
    autoencoder.save(f'autoencoder_{epoch}.pth')


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
    1.  Flow Bytes/s (0.003927684882000904)
    2.  Idle Max (0.00286572429286025)
    3.  Idle Mean (0.0027598217402555747)
    4.  Flow IAT Max (0.0026498937786527677)
    5.  Fwd IAT Max (0.002398798269496183)
    6:  Idle Min (0.00208358678901277)
    7:  Flow Duration (0.001603344715476851)
    8:  Fwd IAT Total (0.0014476983615168137)
    9:  Packet Length Variance (0.001090961768568377)
    10: Bwd IAT Max (0.0008813637814455433)
    11: Fwd IAT Std (0.0008760383625485701)
    12: Bwd IAT Total (0.0007779802235745592)
    13: Flow IAT Std (0.0007390469818346901)
    14: Flow IAT Mean (0.0005638601414830191)
    15: Flow Packets/s (0.0004904000361420913)
    16: Bwd IAT Std (0.0003571593242668314)
    17: Fwd IAT Mean (0.00035411398675933015)
    18: Idle Std (0.00033513264315843116)
    19: Flow IAT Min (0.0002816759206325514)
    20: Bwd IAT Mean (0.00010983653010043781)
    21: Fwd IAT Min (6.229329846973997e-05)
    22: Fwd Packets/s (5.496199264598545e-05)
    23: Bwd IAT Min (3.085926982748788e-05)
    24: Bwd Packet Length Std (2.7228821636526845e-05)
    25: Bwd Packets/s (1.652043101785239e-05)
    26: Bwd Packet Length Max (1.5434392480528913e-05)
    27: Init_Win_bytes_forward (1.3028373359702528e-05)
    28: Active Max (9.772153134690598e-06)
    29: Active Mean (7.83136420068331e-06)
    30: Active Min (7.594657290610485e-06)
    31: Bwd Packet Length Min (7.3010287451324984e-06)
    32: Bwd Packet Length Mean (6.932135875103995e-06)
    33: Avg Bwd Segment Size (6.891992597957142e-06)
    34: Init_Win_bytes_backward (4.881356289843097e-06)
    35: PSH Flag Count (4.112052920390852e-06)
    36: ACK Flag Count (9.614905138732865e-07)
    37: Down/Up Ratio (7.995404303073883e-07)
    38: Subflow Bwd Bytes (7.892813300713897e-07)
    39: Total Length of Bwd Packets (7.873622962506488e-07)
    40: Max Packet Length (6.242389645194635e-07)
    41: Min Packet Length (5.057172529632226e-07)
    42: Packet Length Std (4.336634447099641e-07)
    43: min_seg_size_forward (4.2869214667007327e-07)
    44: Packet Length Mean (4.087924025952816e-07)
    45: Average Packet Size (3.898003342328593e-07)
    46: Fwd Header Length (3.4619915822986513e-07)
    47: Fwd Header Length (3.45518856192939e-07)
    48: FIN Flag Count (3.4434378903824836e-07)
    49: Total Fwd Packets (3.189106791978702e-07)
    50: Subflow Fwd Packets (3.143450157949701e-07)
    51: Total Length of Fwd Packets (3.061068127863109e-07)
    52: Subflow Fwd Bytes (3.0290357244666666e-07)
    53: Total Backward Packets (3.0157025321386755e-07)
    54: Subflow Bwd Packets (3.001605364261195e-07)
    55: Fwd Packet Length Mean (2.9740112950094044e-07)
    56: Avg Fwd Segment Size (2.9278089641593397e-07)
    57: Fwd Packet Length Min (2.898941602325067e-07)
    58: act_data_pkt_fwd (2.8799331630580127e-07)
    59: Fwd Packet Length Max (2.8624708647839725e-07)
    60: Active Std (2.53492544288747e-07)
    61: Bwd Header Length (2.0309016690589488e-07)
    62: RST Flag Count (1.527660060673952e-07)
    63: ECE Flag Count (1.5175282896962017e-07)
    64: SYN Flag Count (8.311872079502791e-08)
    65: Fwd PSH Flags (8.177630661521107e-08)
    66: Bwd PSH Flags (-1.8189894035458565e-12)
    67: Fwd URG Flags (-1.8189894035458565e-12)
    68: Bwd URG Flags (-1.8189894035458565e-12)
    69: CWE Flag Count (-1.8189894035458565e-12)
    70: Fwd Avg Bytes/Bulk (-1.8189894035458565e-12)
    71: Fwd Avg Packets/Bulk (-1.8189894035458565e-12)
    72: Fwd Avg Bulk Rate (-1.8189894035458565e-12)
    73: Bwd Avg Bytes/Bulk (-1.8189894035458565e-12)
    74: Bwd Avg Packets/Bulk (-1.8189894035458565e-12)
    75: Bwd Avg Bulk Rate (-1.8189894035458565e-12)
    76: URG Flag Count (-2.7996065909974277e-08)
    77: Fwd Packet Length Std (-4.727553459815681e-08)

    This labels the top 4 as Flow Bytes/s, Idle Max, Idle Mean, and Flow IAT Max
    instead of the paper's top 4 of Flow Duration, Flow IAT Std, Backward Packet Length Std,
    and Average Package Size.

    The paper's top 4 are labeled the 7th, 13th, 24th, and 45th most important features respectively.

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
    feature_importance: dict[int, tuple[str, int]] = dict()  # Store importance scores for each feature

    for feature_idx in range(num_features):  # Loop over each feature
        print(f'Calculating importance of feature {full_dataset.columns[feature_idx]} ({feature_idx + 1}/{len(full_dataset.columns)})...')
        perturbed_losses = []
        for X, _ in full_dataset:
            if X.dim() == 1:
                X = torch.tensor(np.array([X]))
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
    plt.savefig('feature_importance.png')

    # Print feature importance dict in descending order of importance
    top_features = {key: val for key, val in sorted(feature_importance.items(), key=lambda x: x[1][1], reverse=True)}
    print('Most important features:')
    pprint(top_features, sort_dicts=False)

    # Save json dict of label to feature importance
    with open('feature_idx_to_importance_and_label.json', 'w') as f:
        json.dump(top_features, f)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layers = nn.Sequential(nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 5), nn.ReLU(), nn.Linear(5, 4), nn.ReLU(), nn.Linear(4, 2), nn.ReLU(), nn.Linear(2, 1), nn.Sigmoid())

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


def train_classifier(autoencoder: Autoencoder, classifier: Classifier, train_dataloader: DataLoader, desired_loss=0.0045):
    classifier.train()

    print('\nTraining the classifier...')
    criterion = nn.BCELoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.01)
    loss = torch.tensor(1.0)
    epoch = 0
    while loss.item() > desired_loss:
        for _, (X, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            features = autoencoder.encoder(X)
            preds: torch.Tensor = classifier(features)
            loss: torch.Tensor = criterion(preds, y)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/???], Loss: {loss.item():.7f}')
        epoch += 1
    classifier.save(f'classifier_{epoch}.pth')


def is_ddos(pred: torch.Tensor, threshold: float = 0.5):
    """
    Returns 1 if the prediction is greater than or equal to the threshold, else 0

    threshold should be between 0 and 1
    threshold = 0.5 is equivalent to rounding the prediction to the nearest integer
    """
    return int(pred.item() >= threshold)


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
    # Load the data
    csv_file_path = 'data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
    full_dataset = NetFlowDataSet(csv_file_path)
    train_dataset, test_dataset = random_split(full_dataset, [0.6, 0.4])

    batch_size = 2560
    shuffle = True
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    # Initialize the models
    autoencoder = Autoencoder()
    classifier = Classifier()

    # Train the autoencoder
    # train_autoencoder(autoencoder, train_dataloader)

    # Train the classifier
    # train_classifier(autoencoder, classifier, train_dataloader)

    # Test the models
    autoencoder.load('autoencoder_272.pth')
    classifier.load('classifier_254.pth')
    test_classifier(autoencoder, classifier, full_dataset)

    # Find the most important features
    # find_important_features(autoencoder, full_dataset)


if __name__ == '__main__':
    main()
