from custom_dataset import PvDataset
from simple_CNN import SimpleNet

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms


BATCH_SIZE = 16

train_set = PvDataset("data/pe20180401-20180831_Single.csv")

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                           shuffle=True)
