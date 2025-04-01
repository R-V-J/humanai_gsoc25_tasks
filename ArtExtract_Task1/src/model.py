import torch
import torch.nn as nn
from torchvision import models

class ArtworkCRNN(nn.Module):
    def __init__(self, num_artists, num_genres, num_styles, pretrained=True, rnn_hidden_size=256):
        super(ArtworkCRNN, self).__init__()

        resnet = models.resnet50(pretrained=pretrained)
        self.base_model = nn.Sequential(*list(resnet.children())[:-2])  

        self.feature_dim = 2048  

        self.rnn = nn.GRU(self.feature_dim, rnn_hidden_size, batch_first=True, bidirectional=True)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.artist_classifier = nn.Sequential(
            nn.Linear(rnn_hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_artists)
        )

        self.genre_classifier = nn.Sequential(
            nn.Linear(rnn_hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_genres)
        )

        self.style_classifier = nn.Sequential(
            nn.Linear(rnn_hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_styles)
        )

    def forward(self, x, task='all'):
        x = self.base_model(x) 
        batch_size, channels, h, w = x.shape

        x = x.permute(0, 2, 3, 1) 
        x = x.reshape(batch_size, h * w, channels)  

        rnn_out, _ = self.rnn(x) 

        rnn_out = torch.mean(rnn_out, dim=1)  

        if task == 'all' or task == 'artist':
            artist_out = self.artist_classifier(rnn_out)
        else:
            artist_out = None

        if task == 'all' or task == 'genre':
            genre_out = self.genre_classifier(rnn_out)
        else:
            genre_out = None

        if task == 'all' or task == 'style':
            style_out = self.style_classifier(rnn_out)
        else:
            style_out = None

        return {
            'artist': artist_out,
            'genre': genre_out,
            'style': style_out,
            'features': rnn_out  
        }