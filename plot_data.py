import matplotlib.pyplot as plt
import pandas as pd

for day in range(25, 32):
    daily_data = pd.read_csv("data/08-{}_label.csv".format(day))["power (W)"]
    daily_data.plot()
    plt.xlabel("timestamp")
    plt.ylabel("power (W)")
    plt.title("Power in 2018-08-{}".format(day))
    plt.show()
