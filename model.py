import torch
import torch.nn as nn

from config import SAVE_MODEL_PATH


class CNNWithOneHot(nn.Module):
    def __init__(self, image_width, image_height, character_length, num_classes):
        super(CNNWithOneHot, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Linear((image_width // 8) * (image_height // 8) * 64, 1024),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU())
        self.rfc = nn.Sequential(
            nn.Linear(1024, character_length * num_classes),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.rfc(out)
        return out


def save_model(model, path=SAVE_MODEL_PATH):
    torch.save(model.state_dict(), path)
    print(f"💾 Model saved to: {path}")


def load_model(model_class, path, device):
    model = model_class.to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    print(f"📦 Model loaded from: {path}")
    return model
