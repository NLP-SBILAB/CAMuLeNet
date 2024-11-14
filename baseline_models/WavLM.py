import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

# Load the processor and model
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")
model = WavLMModel.from_pretrained("microsoft/wavlm-base")

# Define the emotion classification model
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(12288, 512)
        self.fc2 = nn.Linear(512, 6)


    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1) 
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Function to extract WavLM features
def extract_features(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)

    # Prepare the input tensors
    inputs = feature_extractor(waveform.squeeze(), sampling_rate=sample_rate, return_tensors="pt", padding=True)

    # Extract features (embeddings) from WavLM
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state.mean(dim=1)

    return outputs