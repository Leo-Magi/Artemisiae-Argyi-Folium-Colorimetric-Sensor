import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from sklearn.preprocessing import LabelEncoder

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.layer4 = nn.Linear(hidden_size, hidden_size)
        self.relu4 = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.layer4(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

data = pd.read_csv('')

X_rgb = data[['Red', 'Green', 'Blue']].values
X_channel = data[['']].values

X_rgb = X_rgb / 255.0

X = np.concatenate((X_rgb, X_channel), axis=1)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['source'].values)

print(y)
time.sleep(5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

model = MLP(input_size=4, hidden_size=4096, num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)

num_epochs = 500
best_accuracy = 0.0
best_model_wts = None

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_wts = model.state_dict()

print(f'Best Accuracy: {best_accuracy:.4f}')
torch.save(best_model_wts, 'best_model2.pth')
print('Best model saved!')

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_accuracy, model):

        score = val_accuracy

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_accuracy, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_accuracy, model)
            self.counter = 0

    def save_checkpoint(self, val_accuracy, model):
        if self.verbose:
            print(f'Validation accuracy increased ({self.best_score:.6f} --> {val_accuracy:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'bestmodel2.pth')

early_stopping = EarlyStopping(patience=10, verbose=True)

num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

    early_stopping(accuracy, model)

    if early_stopping.early_stop:
        print("Early stopping")
        break

print('Best model saved!')