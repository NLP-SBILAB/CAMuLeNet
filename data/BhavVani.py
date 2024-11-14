import numpy as np
import pandas as pd
import os
import random
import pickle
import matplotlib.pyplot as plt
import librosa

from pydub import AudioSegment
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import whisper
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Replace with your file paths
folder_path_mp4 = 'BhavVani/Data_' 
pickle_file_read = 'BhavVani/EncoderOut.pkl'
folder_path = 'BhavVani/Data' 

for filename in os.listdir(folder_path_mp4):
    # Check if the file has a .mp4 extension
    if filename.endswith('.mp4'):
        # Full path to the .mp4 file
        mp4_path = os.path.join(folder_path_mp4, filename)
        # Load the .mp4 file
        audio = AudioSegment.from_file(mp4_path, format="mp4")
        # Define the new .wav file path
        wav_path = os.path.join(folder_path, filename.replace('.mp4', '.wav'))
        # Export as .wav
        audio.export(wav_path, format="wav")
        
        print(f"Converted {filename} to {wav_path}")

print("All .mp4 files have been converted to .wav.")

encoder_outputs_read = {}

with open(pickle_file_read, 'rb') as f1:
    encoder_outputs_read = pickle.load(f1)

# Counter for files processed
i = 0
audio_outputs = {}
# Iterating through files in the folder
for file in os.listdir(folder_path):
    i += 1
    print(i)
    if file.endswith('.wav'):
        # Check in first pickle, then in the second
        encoder_output_ = encoder_outputs_read.get(file)

        if encoder_output_ is not None:
            print(f"Processed file: {file}")
            audio_outputs[file].append(encoder_output_)

print("All .wav files processed from pickle file.")

dict_emotions = {
    "Neutral": 0,
    "Anger": 1,
    "Fear": 2,
    "Surprise": 3,
    "Enjoyment": 6,
    "Contempt": 4,
    'Sadness': 5,
    'Disgust':4
}
df = pd.read_csv('final_annotation.csv')

train_files = []
val_files = []
test_files = []

for file_name in os.listdir(folder_path):
    if file_name.startswith('train_'):
        train_files.append(file_name)
    elif file_name.startswith('val_'):
        val_files.append(file_name)
    elif file_name.startswith('test_'):
        test_files.append(file_name)

# Define a custom dataset class
class MyDataset(Dataset):
    def __init__(self, files, dict_emotions, gender=False, speaker=False):
        self.files = files
        self.dict_emotions = dict_emotions
        self.gender = gender
        self.speaker = speaker

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_file = self.files[idx]
        file_mp4 = file.replace('.wav', '.mp4')
        gender = 1 if df[file_mp4]['gender'] == 'M' else 0
        label = dict[audio_file]
        gender_tensor = torch.tensor(gender, dtype=torch.long).squeeze().to("cuda")
        label_tensor = torch.tensor(label, dtype=torch.long).squeeze().to("cuda")
        encoder_output, mfcc, mel_spec = audio_outputs[audio_file]
        return encoder_output, mfcc, mel_spec, label_tensor, gender_tensor

### UPDATE BATCH SIZE IF NEEDED
batch_size = 64

### Initialise datasets
train_dataset = MyDataset(train_files, dict_emotions)
torch.save(train_dataset, 'train_data.pth')
val_dataset = MyDataset(val_files, dict_emotions)
torch.save(val_dataset, 'val_data.pth')
test_dataset = MyDataset(test_files, dict_emotions)
torch.save(test_dataset, 'test_data.pth')