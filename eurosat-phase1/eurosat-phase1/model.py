import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


def group_norm(channels):
    return nn.GroupNorm(32, channels)


def build_model(num_classes=10, pretrained=True, norm='batch'):
    if norm not in {'batch', 'group'}:
        raise ValueError('norm must be batch or group')
    if pretrained and norm == 'group':
        raise ValueError('Phase 1 GroupNorm comparison must use --from-scratch.')
    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None,
                     norm_layer=nn.BatchNorm2d if norm == 'batch' else group_norm)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = build_model(len(checkpoint['classes']), pretrained=False,
                        norm=checkpoint.get('config', {}).get('norm', 'batch'))
    model.load_state_dict(checkpoint['model_state'])
    return model.to(device).eval(), checkpoint
