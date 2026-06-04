import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Load data
X = np.load("X.npy")
y = np.load("y.npy")

print("Data Loaded")

# Convert to tensor
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Reshape for CNN
X = X.unsqueeze(1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Data Split Done")

# CNN Model
class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=16,
            kernel_size=3
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(16*19, 32)

        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):

        x = self.conv1(x)

        x = self.relu(x)

        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        return x


model = CNNModel()

print("Model Created")

# Loss and optimizer

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# Training

epochs = 20

for epoch in range(epochs):

    optimizer.zero_grad()

    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    loss.backward()

    optimizer.step()

    print("Epoch", epoch+1,
          "Loss:", loss.item())

print("Training Finished")

# Save model

torch.save(
    model.state_dict(),
    "cnn_model.pth"
)

print("Model Saved")