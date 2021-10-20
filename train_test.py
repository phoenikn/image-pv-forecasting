import os

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
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop((1536, 1536)),
        transforms.Resize((1000, 1000))
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    data_dir = "/scratch/itee/uqsxu13"
    training_label_dir = os.path.join(data_dir, "label/training_label_full.csv")
    test_label_dir = os.path.join(data_dir, "label/test_label_full.csv")
    image_dir = os.path.join(data_dir, "images")

    training_label_dir_win = "data/training_label_full.csv"
    test_label_dir_win = "data/test_label_full.csv"
    image_dir_win = "images"

    # Change the directory if run at local
    if os.name == "nt":
        training_label_dir = training_label_dir_win
        test_label_dir = test_label_dir_win
        image_dir = image_dir_win

    train_set = PvDataset(training_label_dir, images_folder=image_dir, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                               shuffle=False)

    test_set = PvDataset(test_label_dir, images_folder=image_dir, transform=transform)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE,
                                              shuffle=False)

    net = SimpleNet()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # The training module
    if not os.path.exists("simpleCNN.pth"):
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

        torch.save(net.state_dict(), "simpleCNN.pth")
        print("Finish training!!!")
    else:
        print("Existed NN")
        net.load_state_dict(torch.load("simpleCNN.pth", map_location=device))
        net.eval()

    mse = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            predicted = outputs[:, 0]
            sub = torch.sub(predicted, labels)
            se = sub.pow(2)
            mse += (torch.sum(se).item() / se.size(0))
            total += se.size(0)
            # print("predicted:", predicted)
            # print("labels:", labels)
            # print("se:", torch.sum(se).item())
            # print("count:", total)

    print("The total number of test data:", total)
    print("MSE is: ", mse)


if __name__ == "__main__":
    main()
