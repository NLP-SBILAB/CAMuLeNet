import random
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

iemocap_pickle_file = "PATH TO PICKLE CONTAINING IEMOCAP STATS"
with open(iemocap_pickle_file, 'rb') as f:
    dump = pickle.load(f)

normal_dict = dict(dump)
list_of_dicts = []
list_of_keys = []

# convert the dictionary to a list of dictionaries by iterating over the keys
for key in normal_dict:
    list_of_dicts.append(dict(normal_dict[key]))
    list_of_keys.append(key)

# Iterate through the list of dicts and for each speaker, save the audio features
speaker_file_names = {}
for i in range(len(list_of_keys)):
    speaker_id = list_of_keys[i]
    speaker_file_names[speaker_id] = []
    speaker_labels = list_of_dicts[i]['seg_label']
    for j in range(len(speaker_labels)):
        file_name = f"{speaker_id}_{speaker_labels[j]}_{j}.wav"
        speaker_file_names[speaker_id].append(file_name)

"""
Unique Speakers and Frequency:
2M 1169
2F 909
3M 1270
3F 983
1M 1187
1F 1037
4M 1033
4F 967
5M 1432
5F 1015
"""
train_files = []
test_files = []
all_files = []

"""
Leave one speaker out cross validation
"""
for speaker in speaker_file_names.keys():
    """
    1M, 4F in test (can be changed to any other speaker pairs)
    """
    if speaker in ['2M', '2F', '3M', '3F', '4M', '1F', '5M', '5F']:
        train_files.extend(speaker_file_names[speaker])
    else:
        test_files.extend(speaker_file_names[speaker])

    all_files.extend(speaker_file_names[speaker])

# Get validation files
val_size = int(0.2 * len(train_files))
val_files = random.sample(train_files, val_size)
train_files = [file for file in train_files if file not in val_files]

print('Number of training files:', len(train_files))
print('Number of validation files:', len(val_files))
print('Number of testing files:', len(test_files))
print('Number of all files:', len(all_files))

"""
Unique Emotions:
"""
dict_emotions = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3
}

encoder_outputs = {}
picke_file_path = "PATH TO PICKLE FILE CONTAINING AUDIO FEATURES and WHISPER EMBEDDINGS"
with open(picke_file_path, 'rb') as f:
    encoder_outputs = pickle.load(f)

"""
Dataset Class for IEMOCAP
"""
class MyDataset(Dataset):
    def __init__(self, files, dict_emotions):
        self.files = files
        self.dict_emotions = dict_emotions

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_file = self.files[idx]
        #print(audio_file)
        gender = 1 if 'M' in audio_file.split("_")[0] else 0
        gender_tensor = torch.tensor(gender, dtype=torch.long).squeeze().to("cuda")
        label = audio_file.split("_")[1]
        label_tensor = torch.tensor([self.dict_emotions[label]], dtype=torch.long).squeeze().to("cuda")

        encoder_output = encoder_outputs[audio_file][0]
        mfcc = encoder_outputs[audio_file][1]
        mel_spec = encoder_outputs[audio_file][2]
        return encoder_output, mfcc, mel_spec, label_tensor, gender_tensor

batch_size = 64 ## Change this to your desired batch size
train_dataset = MyDataset(train_files, dict_emotions)
val_dataset = MyDataset(val_files, dict_emotions)
test_dataset = MyDataset(test_files, dict_emotions)

# Save the datasets
torch.save(train_dataset, 'train_data.pth')
torch.save(val_dataset, 'val_data.pth')
torch.save(test_dataset, 'test_data.pth')