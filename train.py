import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import sys
import os
from model import CAMuLeNet

# Set up command line arguments
parser = argparse.ArgumentParser(description='Train the CAMuLeNet model on emotion and gender recognition tasks in a mutli-task learning setup')
parser.add_argument('--num_emotions', type=int, required=True, help='Number of emotion classes')
parser.add_argument('--data_path', type=str, required=True, help='Path to the data folder containing train, val, and test sets')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
args = parser.parse_args()

# Load data loaders
def load_data_loader(data_path, split):
    data = torch.load(os.path.join(data_path, f'{split}_data.pth'))
    if split == 'train':
        return DataLoader(data, batch_size=args.batch_size, shuffle=True)
    else:
        return DataLoader(data, batch_size=args.batch_size, shuffle=False)

train_loader = load_data_loader(args.data_path, 'train')
val_loader = load_data_loader(args.data_path, 'val')
test_loader = load_data_loader(args.data_path, 'test')

# Define the model
model = CAMuLeNet(number_classes=args.num_emotions)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss function and optimizer
emotion_criterion = nn.CrossEntropyLoss()
gender_criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

# Training and validation function
def train_and_validate(model, train_loader, val_loader, emotion_criterion, gender_criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_train_loss, total_emotion_correct, total_gender_correct, total = 0, 0, 0, 0

        for data in train_loader:
            enc_out, mfcc, mel_spec, emotion_labels, gender_labels = [d.to(device) for d in data]

            optimizer.zero_grad()
            emotion_outputs, gender_outputs = model(enc_out, mfcc, mel_spec)
            emotion_loss = emotion_criterion(emotion_outputs, emotion_labels)
            gender_loss = gender_criterion(gender_outputs, gender_labels)
            total_loss = 0.4 * emotion_loss + 0.1 * gender_loss + 0.2
            total_loss.backward()
            optimizer.step()

            total_train_loss += total_loss.item()
            total += emotion_labels.size(0)
            total_emotion_correct += (emotion_outputs.argmax(1) == emotion_labels).sum().item()
            total_gender_correct += (gender_outputs.argmax(1) == gender_labels).sum().item()

        train_loss = total_train_loss / len(train_loader)
        train_emotion_accuracy = total_emotion_correct / total
        train_gender_accuracy = total_gender_correct / total

        val_loss, val_emotion_accuracy, val_gender_accuracy = validate(model, val_loader, emotion_criterion, gender_criterion)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Emotion Accuracy: {train_emotion_accuracy:.4f}, Train Gender Accuracy: {train_gender_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Emotion Accuracy: {val_emotion_accuracy:.4f}, Val Gender Accuracy: {val_gender_accuracy:.4f}')

def validate(model, val_loader, emotion_criterion, gender_criterion):
    model.eval()
    total_val_loss, total_emotion_correct, total_gender_correct, total = 0, 0, 0, 0
    with torch.no_grad():
        for data in val_loader:
            enc_out, mfcc, mel_spec, emotion_labels, gender_labels = [d.to(device) for d in data]

            emotion_outputs, gender_outputs = model(enc_out, mfcc, mel_spec)
            emotion_loss = emotion_criterion(emotion_outputs, emotion_labels)
            gender_loss = gender_criterion(gender_outputs, gender_labels)
            total_loss = 0.4 * emotion_loss + 0.1 * gender_loss + 0.2

            total_val_loss += total_loss.item()
            total += emotion_labels.size(0)
            total_emotion_correct += (emotion_outputs.argmax(1) == emotion_labels).sum().item()
            total_gender_correct += (gender_outputs.argmax(1) == gender_labels).sum().item()

    val_loss = total_val_loss / len(val_loader)
    val_emotion_accuracy = total_emotion_correct / total
    val_gender_accuracy = total_gender_correct / total
    return val_loss, val_emotion_accuracy, val_gender_accuracy

# Train and validate the model
num_epochs = 10
train_and_validate(model, train_loader, val_loader, emotion_criterion, gender_criterion, optimizer, num_epochs)

# Test function
def test(model, test_loader, emotion_criterion, gender_criterion):
    model.eval()
    total_test_loss, total_emotion_correct, total_gender_correct, total = 0, 0, 0, 0
    with torch.no_grad():
        for data in test_loader:
            enc_out, mfcc, mel_spec, emotion_labels, gender_labels = [d.to(device) for d in data]

            emotion_outputs, gender_outputs = model(enc_out, mfcc, mel_spec)
            emotion_loss = emotion_criterion(emotion_outputs, emotion_labels)
            gender_loss = gender_criterion(gender_outputs, gender_labels)
            total_loss = 0.4 * emotion_loss + 0.1 * gender_loss + 0.2

            total_test_loss += total_loss.item()
            total += emotion_labels.size(0)
            total_emotion_correct += (emotion_outputs.argmax(1) == emotion_labels).sum().item()
            total_gender_correct += (gender_outputs.argmax(1) == gender_labels).sum().item()

    test_loss = total_test_loss / len(test_loader)
    test_emotion_accuracy = total_emotion_correct / total
    test_gender_accuracy = total_gender_correct / total
    print(f'Test Loss: {test_loss:.4f}, Test Emotion Accuracy: {test_emotion_accuracy:.4f}, Test Gender Accuracy: {test_gender_accuracy:.4f}')

# Test the model
test(model, test_loader, emotion_criterion, gender_criterion)
