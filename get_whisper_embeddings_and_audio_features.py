import pickle
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import whisper
import librosa
import numpy as np
import argparse

# Set up command line arguments
parser = argparse.ArgumentParser(description='Get Whisper encoder outputs for your audio dataset')
parser.add_argument('--home_dir', type=str, required=True, help='Path to the home directory')
parser.add_argument('--data', type=str, required=True, help='data name')
parser.add_argument('--model_type', type=str, required=True, help='Whisper model type (base, medium, large)')
parser.add_argument('--split', type=str, required=True, help='Split name (train, val, test)')
parser.add_argument('--r_type', type=int, default=-1, help='Reduction type for encoder output (0: mean, 1: max, 2: min)')
args = parser.parse_args()

# Function to load model
def load_whisper_model(model_type):
    # Load the Whisper model
    model = whisper.load_model(model_type)
    return model

# Loading encoder output
def get_encoder_output(audio_path, model, r_type=-1):
    # Load and preprocess the audio
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # Mel-spectrogram
    mel = whisper.log_mel_spectrogram(audio).to(model.device).unsqueeze(0)

    # Forward pass through the encoder only
    # Note: Whisper's model doesn't provide direct access to the encoder output, so we modify the forward pass.
    with torch.no_grad():
        encoder_output = model.encoder(mel)

    if r_type == 0:
        # take mean across time
        encoder_output = encoder_output.mean(dim=1)
    elif r_type == 1:
        # take max across time
        encoder_output = encoder_output.max(dim=1)[0]
    elif r_type == 2:
        # take min across time
        encoder_output = encoder_output.min(dim=1)[0]
    else:
        # 2D encoder output
        encoder_output = encoder_output

    return encoder_output

whisper_model = load_whisper_model(args.model_type, args.r_type)

# Path to the folder containing audio files
folder_path = f'{args.home_dir}/ADD YOUR PATH HERE'

# Dictionary to store encoder outputs
encoder_outputs = {}

# Initialise the dictionary with a tuple for the file
for file in os.listdir(folder_path):
    if file.endswith('.wav'):
        try:
            file_path = os.path.join(folder_path, file)
            encoder_outputs[file] = []
        except Exception as e:
            print(f"Error processing {file}: {e}")
print("Dictionary Initialised")

# Pickle output file to store encoder outputs
pickle_output_file_path = f'{args.home_dir}/encoder_outputs_{args.model_type}_{args.data}_{args.split}.pkl' 

"""
1. Iterate through the audio files and store Whisper Encoder Outputs
"""
i = 0
print("Model loaded.")
for file in os.listdir(folder_path):
    if file.endswith('.wav'):
        try:
            file_path = os.path.join(folder_path, file)
            large = get_encoder_output(file_path, whisper_model)

            encoder_outputs[file] = large.cpu()
            i+=1
            print(i)
            print(f"Processed {file}")
        except Exception as e:
            print(f"Error processing {file}: {e}")


"""
2. Store MFCCs and Mel-Spectrograms for audio files in the same dictionary
"""
# Get maximum length of audio files
MAX_LEN = 0
for file in os.listdir(folder_path):
    if file.endswith('.wav'):
        audio, sr = librosa.load(os.path.join(folder_path, file), sr=16000)
        MAX_LEN = max(MAX_LEN, len(audio))

print(f"Maximum length: {MAX_LEN}")

# Get MFCCs and Mel-Spectrogram
i = 0
for file in os.listdir(folder_path):
    if file.endswith('.wav'):
        audio, sr = librosa.load(os.path.join(folder_path, file), sr=16000)
        audio = np.pad(audio, (0, MAX_LEN - len(audio)), 'constant')
        
        # Get the mel-spectrogram of the audio
        mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_fft=800, win_length=40, hop_length=10, window="hamming").T
        
        # Expand dimensions to give it 3 channels
        mel = np.expand_dims(mel, axis=2)
        
        # Reshape to C x H x W
        mel = np.reshape(mel, (1, mel.shape[0], mel.shape[1]))
        
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40, hop_length=160, htk=True).T

        #print(f"MFCC: {mfcc.shape}, Mel: {mel.shape}")
        i += 1
        print(i)
        
        encoder_outputs[file].append(mfcc)
        encoder_outputs[file].append(mel)    

# Save all encoder outputs in one pickle file
with open(pickle_output_file_path, 'wb') as f:
    pickle.dump(encoder_outputs, f)   
print("All files processed.")