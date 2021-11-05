# import os
#
# import cv2
import numpy
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd


img1 = Image.open("images/2018-08-25/2018-08-25_10-45-20.jpg")
# img2 = Image.open("images/2018-08-25/2018-08-25_06-02-10.jpg")
ts = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop((1536, 1536)),
        transforms.Resize((224, 224)),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
# images_stack = torch.empty([0, 1536, 2048])
img1 = ts(img1)
plt.imshow(img1.permute(1, 2, 0))
plt.show()
# plt.imshow(img1, cmap="gray")
# red, green, blue = img1.split()
# plt.imshow(red, cmap="Reds")
# plt.show()
# plt.imshow(green, cmap="Greens")
# plt.show()
# plt.imshow(blue, cmap="Blues")
# plt.show()
# img1.show()
# img2 = ts1(img2)
# print(img1.size())
# print(images_stack)
# img_cat = torch.cat((images_stack, img1), 0)
# print(img1)
# print(img_cat.size())
# img_cat = torch.cat((img1, img1), 0)
# print(img_cat.size())

# SIZE = (500, 500)
# new_img = img.resize(SIZE)
# print(new_img.mode)

# batch = torch.rand(16, 3, 10, 10)
# print(batch.size())
# print(not os.path.exists("simpleCNN.pth"))

# x = torch.randint(0, 255, (3, 5, 5))
# x = x.type(torch.DoubleTensor)
# print(x)
# transform = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
# x = transform(x)
# print(x)

# data_dir = "/scratch/itee/uqsxu13"
#
# training_label_dir = os.path.join(data_dir, "label/training_label.csv")
# test_label_dir = os.path.join(data_dir, "label/test_label.csv")
#
# image_dir = os.path.join(data_dir, "images")
#
# print(training_label_dir)
# print(test_label_dir)
# print(image_dir)

# print(os.name == "nt")

# table = pd.read_csv("data/test_label_full.csv")
# power1 = table["power (W)"]
# table = pd.read_csv("data/training_label_full.csv")
# power2 = table["power (W)"]
# table3 = pd.read_csv("data/pe20180401-20180831_Single.csv")
# power3 = table3["power (W)"]

# power1.plot()
# power2.plot()
# power3.plot()
# plt.show()


