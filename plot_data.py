import matplotlib.pyplot as plt
import pandas as pd
import torch

# for day in range(25, 32):
#     daily_data = pd.read_csv("data/08-{}_label.csv".format(day))["power (W)"]
#     daily_data.plot()
#     plt.xlabel("timestamp")
#     plt.ylabel("power (W)")
#     plt.title("Power in 2018-08-{}".format(day))
#     plt.show()

predict = torch.load("predict_result.pt", map_location=torch.device("cpu"))
ground_truth = pd.read_csv("data/test_label_full.csv")["power (W)"]
predict = pd.DataFrame(predict).astype("float")

predict.plot(legend=False)
ground_truth.plot()
plt.show()
