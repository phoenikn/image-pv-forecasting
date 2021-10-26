import pandas as pd

# raw_table = pd.read_csv("data/pe20180401-20180831_FixedWest.csv")

# training_table = raw_table.iloc[124251: 124953]
# training_table = training_table[training_table["power (W)"] != 0]
# test_table = raw_table.iloc[126051: 126756]
# test_table = test_table[test_table["power (W)"] != 0]
# training_table = raw_table.iloc[129600:130400]
# training_table = training_table[training_table["power (W)"] != 0]

# for day in range(25, 32):
#     table = raw_table[raw_table["time"].str.contains("2018-08-{}".format(day))]
#     table = table[table["power (W)"] != 0]
#     table.to_csv("data/extracted/08-{}_west.csv".format(day), index=False)
#
# print("Finish!")

# sunny_tables = [pd.read_csv("data/08-29_label.csv"), pd.read_csv("data/08-30_label.csv"),
#                 pd.read_csv("data/08-31_label.csv")]
# sunny_tables = pd.concat(sunny_tables)
# sunny_tables.to_csv("data/sunny_train_three.csv", index=False)

# raw_table = pd.read_csv("data/HiResGatton_20180818-0830.csv")
#
# for day in range(25, 32):
#     table = raw_table[raw_table["DateTime"].str.contains("{}/08/2018".format(day))]
#     table = table[table["power (kW)"] < 0]
#     table.to_csv("data/08-{}_hires.csv".format(day), index=False)
#
# print("Finish!")


# training_table.to_csv("data/training_label_full.csv", index=False)
# test_table.to_csv("data/test_label_full.csv", index=False)
