import torch
import torch.nn as nn
import librosa
import numpy as np


class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv1d(1,16,3)

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(16*19,32)

        self.fc2 = nn.Linear(32,2)

    def forward(self,x):

        x=self.conv1(x)

        x=self.relu(x)

        x=self.pool(x)

        x=x.view(x.size(0),-1)

        x=self.fc1(x)

        x=self.relu(x)

        x=self.fc2(x)

        return x


model=CNNModel()

model.load_state_dict(
    torch.load("cnn_model.pth")
)

model.eval()

print("Model Loaded")

# File path (any format)
file_path = "bha.ogg"   # change if needed

import soundfile as sf

# Load any format
audio, sr = librosa.load(file_path, sr=None)

# Convert to WAV
sf.write("converted.wav", audio, sr)

# Reload WAV
audio, sr = librosa.load("converted.wav", sr=None)

# Extract MFCC (THIS WAS MISSING)
mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=40
)

# Take mean
mfcc = np.mean(mfcc.T, axis=0)

mfcc=torch.tensor(
    mfcc,
    dtype=torch.float32
)

mfcc=mfcc.unsqueeze(0)

mfcc=mfcc.unsqueeze(0)

import torch.nn.functional as F

with torch.no_grad():
    output = model(mfcc)

# Apply softmax
probs = F.softmax(output, dim=1)

# Get prediction
confidence, prediction = torch.max(probs, 1)

confidence = confidence.item() * 100
prediction = prediction.item()

if prediction == 0:
    print(f"Real Audio ({confidence:.2f}% confidence)")
else:
    print(f"Fake Audio ({confidence:.2f}% confidence)")