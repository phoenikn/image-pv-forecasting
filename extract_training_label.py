import pandas as pd

raw_table = pd.read_csv("data/pe20180401-20180831_Single.csv")

training_table = raw_table.iloc[124251: 124953]

# training_table.to_csv("data/training_label.csv", index=False)
