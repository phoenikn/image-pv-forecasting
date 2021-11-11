# import os
#
# import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
# import torch
# import torchvision.transforms as transforms
import pandas as pd
import time

start = time.time()
img1 = Image.open("images/2018-08-25/2018-08-25_10-31-10.jpg").convert("L")
# img2 = Image.open("images/2018-08-25/2018-08-25_06-02-10.jpg")
# ts = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.CenterCrop((1536, 1536)),
#         transforms.Resize((224, 224)),
#         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
# ])
# images_stack = torch.empty([0, 1536, 2048])
img1 = img1.crop((256, 0, 1792, 1536))
plt.imshow(img1, cmap="gray")
plt.show()
im_arr = np.asarray(img1)
print(im_arr.shape)
# im_arr_reshape = im_arr.reshape(im_arr.shape[0] * im_arr.shape[1], im_arr.shape[2])
im_arr_reshape = im_arr.reshape(im_arr.shape[0] * im_arr.shape[1], 1)
im_arr_reshape = MinMaxScaler().fit_transform(im_arr_reshape)
kmeans = KMeans(n_clusters=5).fit(im_arr_reshape)
print(kmeans.labels_.shape)
im_cluster_arr = kmeans.labels_.reshape(1536, 1536)
im_cluster_arr = im_cluster_arr * (256/4) - 1
im_cluster = Image.fromarray(im_cluster_arr)
plt.imshow(im_cluster)
plt.show()

end = time.time()
print(end - start)
# plt.imshow(img1, cmap="gray")
# plt.show()
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


