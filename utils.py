import torch
import torch.nn as nn
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def load_model(path):
    model = SimpleCNN()
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model


def predict_regime(model, data):
    tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    output = model(tensor)
    probabilities = torch.softmax(output, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][prediction].item() * 100
    return prediction, confidence
