import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        layer.bias.data.fill_(0.)

def init_bn(bn):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class SER_AlexNet(nn.Module):
    def __init__(self, num_classes=4, in_ch=3, pretrained=True):
        super(SER_AlexNet, self).__init__()
        model = torchvision.models.alexnet(pretrained=pretrained)
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = model.classifier
        if in_ch != 3:
            self.features[0] = nn.Conv2d(in_ch, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
            init_layer(self.features[0])
        self.classifier[6] = nn.Linear(4096, num_classes)
        self._init_weights(pretrained=pretrained)
        print('\n<< SER AlexNet Finetuning model initialized >>\n')

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x_ = torch.flatten(x, 1)
        out = self.classifier(x_)
        return x, out

    def _init_weights(self, pretrained):
        init_layer(self.classifier[6])
        if not pretrained:
            for i in [0, 3, 6, 8, 10]:
                init_layer(self.features[i])
            for i in [1, 4]:
                init_layer(self.classifier[i])

# Define the co-attention based classifier - CAMuLeNet
class CAMuLeNet(nn.Module):
    def __init__(self, number_classes, number_genders=2):
        super(CAMuLeNet, self).__init__()
        self.num_classes = number_classes
        self.num_genders = number_genders

        # Alexnet for Spectrogram
        self.alexnet = SER_AlexNet(num_classes=number_classes, in_ch=1, pretrained=True)
        self.post_alexnet_dropout = nn.Dropout(0.1)
        self.post_alexnet_fc = nn.Linear(9216, 1024)

        # Bidirectional GRU for MFCCs
        self.gru_mfcc = nn.GRU(input_size=40, hidden_size=256, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.post_gru_mfcc_dropout = nn.Dropout(0.1)
        self.post_gru_mfcc_fc = nn.Linear(256512, 1024)

        # Layer for MFCC + Spectrogram concatenation
        self.post_concat_dropout = nn.Dropout(0.1)
        self.post_concat_fc = nn.Linear(2048, 1500)

        # Whisper Layers
        self.whisper_dropout = nn.Dropout(0.1)
        self.whisper_fc = nn.Linear(1024, 1024)

        # Combined embeddings
        self.post_attention_dropout = nn.Dropout(0.1)
        self.post_attention_fc1 = nn.Linear(3072, 128)
        self.post_attention_fc2 = nn.Linear(128, 128)
        
        # Final output layer for emotion
        self.output_fc_emotion = nn.Linear(128, number_classes)

        # Final output layer for gender 
        self.output_fc_gender = nn.Linear(128, number_genders)
        self.combined_embeddings = None


    def forward(self, ptm_output, mfcc, mel_spec):
        # Forward Pass through the AlexNet for the Spectrogram
        alexnet_features, _ = self.alexnet(mel_spec) 
        alexnet_features = alexnet_features.view(alexnet_features.size(0), -1) 

        # Dropout and FC for the AlexNet output
        alexnet_post_drop = self.post_alexnet_dropout(alexnet_features) 
        alexnet_post_fc = F.relu(self.post_alexnet_fc(alexnet_post_drop), inplace=False) 

        # Normalise the MFCCs
        mfcc = F.normalize(mfcc, p=2, dim=1)
        mfcc_post_gru, _ = self.gru_mfcc(mfcc) 

        # Flatten the Bi-GRU output
        mfcc_post_gru = torch.flatten(mfcc_post_gru, start_dim=1) 

        # Dropout and FC for the MFCC output
        mfcc_post_drop = self.post_gru_mfcc_dropout(mfcc_post_gru)
        mfcc_post_fc = F.relu(self.post_gru_mfcc_fc(mfcc_post_drop), inplace=False) 

        # Concatenate the MFCC and Spectrogram outputs
        concat = torch.cat((alexnet_post_fc, mfcc_post_fc), dim=1) 
        concat_post_drop = self.post_concat_dropout(concat)
        concat_post_fc = F.relu(self.post_concat_fc(concat_post_drop), inplace=False) 
        concat_post_fc = concat_post_fc.unsqueeze(1) 

        # Remove the extra dimension from the whisper2 output
        ptm_output = ptm_output.squeeze(1) 

        # Dropout and FC for the whisper2 output
        whisper_post_att = torch.matmul(concat_post_fc, ptm_output) 
        whisper_post_att = whisper_post_att.reshape(whisper_post_att.size(0), -1) 

        ptm_output_post_drop = self.whisper_dropout(whisper_post_att) 
        ptm_output_post_fc = F.relu(self.whisper_fc(ptm_output_post_drop), inplace=False) 

        # Combine mfcc, spectrogram and co-attention outputs
        combined_embeddings = torch.cat((mfcc_post_fc, alexnet_post_fc, ptm_output_post_fc), dim=1) 
        self.combined_embeddings = self.post_attention_dropout(combined_embeddings) 
        combined_embeddings_post_fc1 = F.relu(self.post_attention_fc1(self.combined_embeddings), inplace=False) 
        combined_embeddings_post_fc2 = F.relu(self.post_attention_fc2(combined_embeddings_post_fc1), inplace=False) 
        
        # Final output layers for emotion and gender
        output_embeddings_emotion = self.output_fc_emotion(combined_embeddings_post_fc2)
        output_embeddings_gender = self.output_fc_gender(combined_embeddings_post_fc2)

        return output_embeddings_emotion, output_embeddings_gender