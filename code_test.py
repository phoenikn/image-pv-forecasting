import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

img1 = Image.open("images/2018-08-25/2018-08-25_06-02-00.jpg")
# img2 = Image.open("images/2018-08-25/2018-08-25_06-02-10.jpg")
# ts = transforms.Compose([
#     transforms.ToTensor()
# ])
# images_stack = torch.empty([0, 1536, 2048])
# img1 = ts(img1)
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
