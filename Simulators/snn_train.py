import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch
from snntorch import surrogate
import snntorch as snn
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
import os

# Adjust the Python path to include the spkeras module
currentpath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(currentpath, '../spkeras'))


# Load the dataset
data = pd.read_csv('training_data.csv')

# Extract features and target
X = data[['theta', 'omega', 'velocity', 'targetvelocity', 'x_position']].values
y = data['acc'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Create DataLoader for training
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the spiking neural network
class SNN(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()
        self.fc1 = nn.Linear(5, 128)
        self.lif1 = snn.Leaky(
            beta=0.9, spike_grad=surrogate.fast_sigmoid())
        self.fc2 = nn.Linear(128, 64)
        self.lif2 = snn.Leaky(
            beta=0.9, spike_grad=surrogate.fast_sigmoid())
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)
        out = self.fc3(spk2)
        return out

# Initialize the model, loss function, and optimizer
model = SNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Save the trained model weights
torch.save(model.state_dict(), 'snn_model_weights.pth')
