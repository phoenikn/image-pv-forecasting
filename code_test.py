# date = "2018-08-25"
# time = "00:00:00"
# time = time.replace(":", "-")
# if int(time[3:5]) == 0:
#     if int(time[0:2]) == 0:
#         time = "23-59" + time[5:]
#         # The special condition that the date is
#         # the first day of the month is ignored, able to add if needed
#         date = date[0:-2] + str(int(date[-2:]) - 1)
#     else:
#         time = "{:02d}".format(int(time[0:2]) - 1) + "-59" + time[5:]
# else:
#     time = time[0:3] + "{:02d}".format(int(time[3:5]) - 1) + time[5:]
#
# print(date)
# print(time)
