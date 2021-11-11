import numpy as np
import pandas as pd
from PIL import Image
import os

directory = "images/American/07"
grayscale = []
i = 0
start = False
end = False
for filename in os.listdir(directory):
    # if "06-36" in filename:
    #     start = True
    # if "17-19" in filename:
    #     end = True
    # if start:
    img = Image.open(os.path.join(directory, filename)).convert("L")
    img_array = np.asarray(img)
    gray_mean = img_array.sum() / img_array.size
    grayscale.append(gray_mean)
    i += 1
    print(i)
    # if end:
    #     break

pd.DataFrame(grayscale, columns=["grayscale"]).to_csv("data/extracted/US_0107_gray.csv", index=False)
print("Finish!")
