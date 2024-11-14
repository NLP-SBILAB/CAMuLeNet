import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Initialize wav2vec components
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

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
    
def extract_features(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)

    # Resample the waveform to 16kHz if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Wav2Vec2Processor expects a 1D array (shape: [sequence_length]), but torchaudio.load returns a 2D array (shape: [num_channels, sequence_length])
    # Therefore, we take the first channel only if it's stereo (2 channels)
    if waveform.shape[0] > 1:
        waveform = waveform[0]

    input_values = processor(waveform, return_tensors="pt").input_values

    # Ensure the input is 2D: [batch_size, sequence_length]
    if input_values.dim() == 3:
        input_values = input_values.squeeze(0)

    with torch.no_grad():
        hidden_states = model(input_values).last_hidden_state

    return hidden_states

