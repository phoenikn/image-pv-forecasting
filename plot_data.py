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

# gr = pd.read_csv("data/extracted/US_0107_gray.csv")["grayscale"]
# gr.plot(label="grayscale")
# plt.legend()
# plt.show()

hires = pd.read_csv("data/PV Data Hires/20180701-0712.csv")
hi_01 = hires[hires["DateTime"].str.startswith("1/07")]
hi_01_p = hi_01["AQG1_B001_PM001.Sts.P_kW"]
hi_01_p = hi_01_p * -1
hi_01_p.plot()
plt.xlabel("timestamp")
plt.ylabel("power")
plt.show()

# predict = torch.load("predict_result.pt", map_location=torch.device("cpu"))
# ground_truth = pd.read_csv("data/extracted/test_label_norm.csv")["power (W)"]
# predict = pd.DataFrame(predict).astype("float")
#
# predict.plot(legend=False)
# ground_truth.plot()
# plt.show()

