import matplotlib.pyplot as plt
import pandas as pd
# import torch


def plot_daily(panel: str):
    for day in range(25, 32):
        if panel == "hires":
            hires = pd.read_csv("data/extracted/08-{}_hires.csv".format(day))["power (kW)"]
            hires *= -1
            hires.plot()
        else:
            low_res = pd.read_csv("data/extracted/08-{}_{}.csv".format(day, panel))["power (W)"]
            low_res.plot()

        plt.xlabel("timestamp")
        plt.ylabel("power")
        plt.title("Power in 2018-08-{}".format(day))
        plt.show()


# plot_daily(input("Which panel?"))

# predict = torch.load("predict_result.pt", map_location=torch.device("cpu"))
ground_truth = pd.read_csv("data/extracted/sunny_train_three_norm.csv")["power (W)"]
# predict = pd.DataFrame(predict).astype("float")

# predict.plot(legend=False)
ground_truth.plot()
plt.show()

# train_set_norm = pd.read_csv("data/extracted/test_label_norm.csv")["power (W)"]
# train_set_norm.plot()
# plt.show()
