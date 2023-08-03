import torch.nn as nn
import pickle

class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out
    
    def save_weights(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.state_dict(), f)

    @staticmethod
    def load_weights(filepath, num_of_layers, channels=1):
        model = DnCNN(channels, num_of_layers)
        with open(filepath, 'rb') as f:
            state_dict = pickle.load(f)
        model.load_state_dict(state_dict)
        return model
