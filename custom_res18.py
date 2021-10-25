import torch
import torch.nn as nn
import torchvision.models as models

backbone = models.resnet18(pretrained=False)
print(backbone.conv1)