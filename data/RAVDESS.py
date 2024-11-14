from enum import unique
import os
import pickle
import random
from torch.utils.data import Dataset, DataLoader
import torch

import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torchaudio

# Load the pickle file with APPRORIATE PATH
pickle_file = "ADD PATH TO PICKLE FILE"
encoder_outputs = {}
with open(pickle_file, 'rb') as file:
    encoder_outputs =  pickle.load(file)

# Modify the folder path
folder_path = "RAVDESS"

audio_outputs = {}
i = 0
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.wav'):
            path = os.path.join(root, file)
            audio_outputs[file] = []  # Initialize if additional processing needed
            if file in encoder_outputs:
                audio_outputs[file].append(encoder_outputs[file])
            i += 1
            print(i)

print("All .wav files processed from pickle file.")

dict_emotions = {
    '01': 0, '02': 1, '03': 2, '04': 3, '05': 4,
    '06': 5, '07': 6, '08': 7
}

# Split files into training, val and test sets
files = list(audio_outputs.keys())
unique_numbers = set()
# Randomly sampliing 2 speakers from the files in the dataset
for file in files:
    number = file.split("-")[-1].replace('.wav', '')
    unique_numbers.add(number)
if len(unique_numbers) >= 2:
    selected_numbers = random.sample(unique_numbers, 2)
else:
    selected_numbers = list(unique_numbers)
test_files = [file for file in files if file.split("-")[-1].replace('.wav', '') in selected_numbers]
train_files = [file for file in files if file not in test_files]
val_ratio = 0.2  # 20% of the training files for validation
val_size = int(len(train_files) * val_ratio) 
val_files = random.sample(train_files, val_size)
train_files = [file for file in train_files if file not in val_files]

# Define a custom dataset class
class MyDataset(Dataset):
    def __init__(self, files, dict_emotions, speaker=False, gender=False):
        self.files = files
        self.dict_emotions = dict_emotions
        self.speaker = speaker
        self.gender = gender

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_file = self.files[idx]
        label = audio_file.split("-")[2]
        label_tensor = torch.tensor([self.dict_emotions[label]], dtype=torch.long).squeeze().to("cuda")
        gender_ = audio_file.split("-")[-1]
        if int(gender_.split(".")[0])%2 == 0:
            gender = 1
        else: 
            gender = 0
        
        gender_tensor = torch.tensor(gender, dtype=torch.long).squeeze().to("cuda")
        encoder_output, mfcc, mel_spec = audio_outputs[audio_file]
        return encoder_output, mfcc, mel_spec, label_tensor, gender_tensor

batch_size = 64 # Change this to your desired batch size
train_dataset = MyDataset(train_files, dict_emotions)
val_dataset = MyDataset(val_files, dict_emotions)
test_dataset = MyDataset(test_files, dict_emotions)

# Save the datasets
torch.save(train_dataset, 'train_data.pth')
torch.save(val_dataset, 'val_data.pth')
torch.save(test_dataset, 'test_data.pth')
