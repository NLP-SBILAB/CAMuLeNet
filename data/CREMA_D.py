import os
import shutil
import random
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import pickle

# Paths
folder_path = 'CREMA-D/AudioWAV'  # Update with the actual path if different
root_wav = folder_path
train_root = "TrainAudioWAV"
val_root = "ValAudioWAV"
test_root = "TestAudioWAV"

dict_emotions = {
    "ANG": 0,
    "DIS": 1,
    "FEA": 2,
    "HAP": 3,
    "NEU": 4,
    "SAD": 5
}

# Ensure directories exist
for directory in [train_root, val_root, test_root]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Define Test Speakers and Gather Files for Training/Validation
speaker_ids = []

# Collect all speaker IDs by splitting filenames
for file in os.listdir(folder_path):
    speaker_ids.append(file.split("_")[0])

# Get unique speaker IDs and take every second one up to the number 'n'
unique_speakers = list(set(speaker_ids))
"""
Leave-one-speaker-out cross validation
(can be modified through random sampling or other techniques)
"""
n = 9  
test_speakers = [unique_speakers[i] for i in range(0, len(unique_speakers), 2)][:n]

# Separate files into test and non-test
test_files = [file for file in os.listdir(root_wav) if file.split("_")[0] in test_speakers]
files_val_train = [file for file in os.listdir(root_wav) if file.split("_")[0] not in test_speakers]

# Shuffle and Split into Training and Validation Sets
random.shuffle(files_val_train)
val_files = files_val_train[:372]  # Taking 372 files for validation
train_files = files_val_train[372:]  # Remaining files for training

# Print split sizes
print(f"Validation set has {len(val_files)} files.")
print(f"Training set has {len(train_files)} files.")
print(f"Test set has {len(test_files)} files.")

# Copy Files to Respective Folders
def copy_files(file_list, source_folder, destination_folder):
    for file in file_list:
        shutil.copy(os.path.join(source_folder, file), os.path.join(destination_folder, file))

# Copy files to train, validation, and test folders
copy_files(train_files, root_wav, train_root)
copy_files(val_files, root_wav, val_root)
copy_files(test_files, root_wav, test_root)

print("Files copied successfully to their respective folders.")

# Encoder outputs
pickle_file_read = 'Medium_CoAtt.pkl' # Modify file path

# Loading encoder outputs, MFCCs and Mel Spectrogram from pickle file into memory
encoder_outputs_read = {}
with open(pickle_file_read, 'rb') as f1:
    encoder_outputs_read = pickle.load(f1)

# Counter for files processed
i = 0
# Dictionary to store read encoder outputs
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

# Replace the file path
df  = pd.read_csv("CREMA-D/VideoDemographics.csv")
# Define a custom dataset class
class MyDataset(Dataset):
    def __init__(self, files, dict_emotions, gender=False, speaker=False):
        self.files = files
        self.dict_emotions = dict_emotions
        self.speaker = speaker
        self.gender = gender

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_file = self.files[idx]
        #print(audio_file)
        label = audio_file.split("_")[2]
        actor_id = audio_file.split("_")[0]
        gender_ = df[df['ActorID'] == int(actor_id)]['Sex'].iloc[0]
        if gender_ == 'Male':
            gender = 1
        else:
            gender = 0
        gender_tensor = torch.tensor(gender, dtype=torch.long).squeeze().to("cuda")
        label_tensor = torch.tensor([self.dict_emotions[label]], dtype=torch.long).squeeze().to("cuda")
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