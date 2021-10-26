import os

from custom_dataset import PvDataset
from simple_CNN import SimpleNet

import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import math
from custom_res18 import resnet18

BATCH_SIZE = 16


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop((1536, 1536)),
        # transforms.Resize((1000, 1000)),
        transforms.Resize((224, 224))
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    data_dir = "/scratch/itee/uqsxu13"
    training_label_dir = os.path.join(data_dir, "label/sunny_train_three.csv")
    test_label_dir = os.path.join(data_dir, "label/test_label_full.csv")
    image_dir = os.path.join(data_dir, "images")

    training_label_dir_win = "data/extracted/sunny_train_three.csv"
    test_label_dir_win = "data/extracted/test_label_full.csv"
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

    # net = SimpleNet()
    net = resnet18()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # The training module
    if not os.path.exists("simpleCNN.pth"):
        for epoch in range(2):
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                labels = labels.float()

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        torch.save(net.state_dict(), "simpleCNN.pth")
        print("Finish training!!!")
    else:
        print("Existed NN")
        net.load_state_dict(torch.load("simpleCNN.pth", map_location=device))
        net.eval()

    print("Device is:", device)
    total_se = 0
    total = 0
    observed_sum = 0

    predict_result = torch.tensor([]).to(device)

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            predicted = outputs[:, 0]
            sub = torch.sub(predicted, labels)
            se = sub.pow(2)
            total_se += torch.sum(se).item()
            total += se.size(0)
            observed_sum += torch.sum(labels).item()

            predict_result = torch.cat((predict_result, predicted))
            # print("predicted:", predicted)
            # print("labels:", labels)
            # print("se:", torch.sum(se).item())
            # print("count:", total)

    torch.save(predict_result, "predict_result.pt")

    # MSE: Mean Square Error
    # RMSE: Root Mean Square Error
    # nRMSE: Normalized Root Mean Square Error
    mse = total_se/total
    rmse = math.sqrt(mse)
    nrmse = rmse/(observed_sum/total)
    print("The total number of test data:", total)
    print("MSE is:", mse)
    print("RMSE is:", rmse)
    print("nRMSE is:", nrmse)


if __name__ == "__main__":
    main()
