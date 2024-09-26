"""
Using reference code found here:
https://b-nova.com/en/home/content/anomaly-detection-with-random-forest-and-pytorch/
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, random_split
from time import perf_counter

import csv


class NetFlowDataSet(Dataset):
    def __init__(self, csv_file):
        with open(csv_file, 'r') as f:
            self.data = list(map(lambda row: [row[7], row[19], row[23], row[55], row[-1]], list(csv.reader(f))[1:]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        x1 = row[0]  # Flow Duration
        x2 = row[1]  # Bwd Packet Length Std
        x3 = row[2]  # Flow IAT Std
        x4 = row[3]  # Avg Packet Size

        xs = [x1, x2, x3, x4]
        values = np.array(xs, dtype=np.float32)  # 4 features
        is_ddos = row[-1] == 'DDoS'
        X = values
        y = int(is_ddos)
        return X, y


# Load the dataset
data_path = 'data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
dataset = NetFlowDataSet(data_path)
train_dataset, test_dataset = random_split(dataset, [0.6, 0.4])

X_train, y_train = [xval for xval, _ in train_dataset], [yval for _, yval in train_dataset]
X_test, y_test = [xval for xval, _ in test_dataset], [yval for _, yval in test_dataset]

# Creating a Random Forest classifier object
# Arbitrarily set n_estimators to 120, random_state to ~42% of that
rfc = RandomForestClassifier(n_estimators=120, random_state=50)

# Fitting the Random Forest classifier to the training data
start = perf_counter()
rfc.fit(X_train, y_train)
end = perf_counter()
print(f'Training time: {(end - start)}s')

# Making predictions on the testing data
y_pred = rfc.predict(X_test + X_train)

# Printing the confusion matrix and classification report
cm = confusion_matrix(y_test + y_train, y_pred)
print(cm)
print(classification_report(y_test + y_train, y_pred))

# Plotting the confusion matrix
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix')
plt.show()
