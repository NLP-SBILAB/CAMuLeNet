from json import encoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import Wav2Vec2Processor, Wav2Vec2Model

folder_path = 'CaFE/audio' # replace with path to audio files 
dict_emotions = {
    "C": 0,
    "J": 1,
    "D": 2,
    "N": 3,
    "S": 4,
    "P": 5,
    "T": 6
}

# id: (gender, age)
speaker_info = {
    1: ("M", 46),
    2: ("F", 64),
    3: ("M", 18),
    4: ("F", 50),
    5: ("M", 22),
    6: ("F", 34),
    7: ("M", 15),
    8: ("F", 25),
    9: ("M", 42),
    10: ("F", 20),
    11: ("M", 35),
    12: ("F", 37)
}

speakers = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

def pick_one_odd_one_even(speakers):
    odd_list = [speaker for speaker in speakers if int(speaker) % 2 != 0]
    even_list = [speaker for speaker in speakers if int(speaker) % 2 == 0]
    chosen_odd = random.choice(odd_list)  
    chosen_even = random.choice(even_list) 
    return [chosen_odd, chosen_even]

test_speakers = pick_one_odd_one_even(speakers)
test_files = []
train_files = []
for file in os.listdir(folder_path):
    if file.split('-')[0] in test_speakers:
        test_files.append(file)
    else:
        train_files.append(file)
val_size = int(0.2 * len(train_files))
val_files = random.sample(train_files, val_size)
train_files = [file for file in train_files if file not in val_files]

"""
Path to pickle file containing audio features and whisper embeddings
"""
pickle_file_read = '"ADD PATH TO PICKLE FILE"'
encoder_outputs_read = {}

with open(pickle_file_read, 'rb') as f:
    encoder_outputs_read = pickle.load(f)
i = 0
audio_outputs = {}
for file in os.listdir(folder_path):
    if file.endswith(".wav"):
        audio_outputs[file] = encoder_outputs_read[i]
        i += 1
        print(i)
        encoder_output_ = encoder_outputs_read.get(file)
        if encoder_output_ is not None:
            print(f"Processed file: {file}")
            audio_outputs[file] = encoder_output_

print("All .wav files processed from pickle file.")
# Define a custom dataset class
class MyDataset(Dataset):
    def __init__(self, files, dict_emotions, speaker = False, gender = False):
        self.files = files
        self.dict_emotions = dict_emotions
        self.speaker = speaker
        self.gender = gender

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_file = self.files[idx]
        gender = 1 if speaker_info[int(audio_file.split("-")[0])][0] == 'M' else 0
        gender_tensor = torch.tensor(gender, dtype=torch.long).squeeze().to("cuda")
        label = audio_file.split("-")[1]
        label_tensor = torch.tensor([self.dict_emotions[label]], dtype=torch.long).squeeze().to("cuda")

        encoder_output, mfcc, mel_spec = audio_outputs[audio_file]
        return encoder_output, mfcc, mel_spec, label_tensor, gender_tensor

batch_size = 64
train_dataset = MyDataset(train_files, dict_emotions)
val_dataset = MyDataset(val_files, dict_emotions)
test_dataset = MyDataset(test_files, dict_emotions)

# Save the datasets
torch.save(train_dataset, 'train_data.pth')
torch.save(val_dataset, 'val_data.pth')
torch.save(test_dataset, 'test_data.pth')