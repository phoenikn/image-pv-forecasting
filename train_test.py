from custom_dataset import PvDataset
from simple_CNN import SimpleNet

import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import numpy as np

BATCH_SIZE = 16


def main():
    norm_channels = np.full(
        shape=3,
        fill_value=0.5,
        dtype=np.float
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((1000, 1000)),
        transforms.Normalize(mean=norm_channels, std=norm_channels)
    ])

    train_set = PvDataset("data/training_label.csv", transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                               shuffle=False)

    test_set = PvDataset("data/test_label.csv", transform=transform)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE,
                                              shuffle=False)

    net = SimpleNet()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            labels = labels.float()

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0

    print("Finish!!!")

    torch.save(net.state_dict(), "simpleCNN.pth")

    mse = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            mse += ((labels - predicted) * (labels - predicted))

    print("MSE is: " + str(mse / total))


if __name__ == "__main__":
    main()
