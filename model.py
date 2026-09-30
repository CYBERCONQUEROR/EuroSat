import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


def build_model(num_classes=10, pretrained=True):
    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model  # Raw logits; CrossEntropyLoss handles log-softmax internally.


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = build_model(len(checkpoint['classes']), pretrained=False)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device).eval()
    return model, checkpoint
