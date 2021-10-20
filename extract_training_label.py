import pandas as pd

raw_table = pd.read_csv("data/pe20180401-20180831_Single.csv")

training_table = raw_table.iloc[124251: 124953]
training_table = training_table[training_table["power (W)"] != 0]
test_table = raw_table.iloc[126051: 126756]
test_table = test_table[test_table["power (W)"] != 0]

# training_table.to_csv("data/training_label_full.csv", index=False)
# test_table.to_csv("data/test_label_full.csv", index=False)
