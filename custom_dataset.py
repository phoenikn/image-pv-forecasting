import os
import torch.utils.data
import cv2
import pandas as pd


class PvDataset(torch.utils.data.Dataset):
    """
    The PV datasets, including the all-sky pictures and correlated power and energy
    """

    def __init__(self, csv_path, images_folder="images", transform=None):
        """

        :param csv_path (string): Path of the csv for labels
        :param images_folder (string): Path of the image polder
        :param transform (transform): Optional transform
        """
        self.df = pd.read_csv(csv_path)
        self.images_folder = images_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        date, time = self.df.iloc[index, 0].split()

        # Select the timestamp of the last minute
        time = time.replace(":", "-")
        # If minute is 00
        if int(time[3:5]) == 0:

            # If hour and minute are both 00
            if int(time[0:2]) == 0:
                time = "23-59" + time[5:]
                # The special condition that the date is
                # the first day of the month is ignored,
                # able to add if needed
                date = date[0:-2] + str(int(date[-2:]) - 1)
            else:
                time = "{:02d}".format(int(time[0:2]) - 1) + "-59" + time[5:]

        else:
            time = time[0:3] + "{:02d}".format(int(time[3:5]) - 1) + time[5:]

        # Read the pictures per 10 seconds of last 60 seconds
        images = []
        for second in range(0, 60, 10):
            time = time[0:-2] + "{:02d}".format(second)
            img_path = os.path.join("images", date, date + "_" + time + ".jpg")
            image = cv2.imread(img_path)
            images.append(image)

        label = int(self.df.iloc[index, 1])

        # Return the six pictures of the last minute and the current power as label
        return images, label
