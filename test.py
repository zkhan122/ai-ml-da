import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("D:\\programs\\ai-ml-da\\pandas-learn\\datasets\\avocado.csv")
df = df.copy()[ df["type"] == "organic"]
df["Date"] = pd.to_datetime( df["Date"])

df.sort_values(by="Date", ascending=True, inplace=True)

print(df.head())

graph_df = pd.DataFrame()

for region in df["region"].unique():
    region_df = df.copy()[ df["region"] == region]
    region_df.set_index("Date", inplace=True)
    region_df.sort_index(inplace=True)
    region_df[f"{region}_PricePer25MA"] = region_df["AveragePrice"].rolling(25).mean()
    region_df.dropna(inplace=True)
    if graph_df.empty:
        graph_df = region_df[[f"{region}_PricePer25MA"]]
    else:
        graph_df = graph_df.join(region_df[f"{region}_PricePer25MA"])

graph_df.dropna(inplace=True)
graph_df.tail()

arr = graph_df.values
print("Buffer: ", arr)

arr = arr[~np.isnan(arr)] # not nan

print()

# print(graph_df.plot(figsize=(15, 5), legend=False))

# set plt attributes
plt.rcParams["figure.figsize"] = [15, 4]
plt.rcParams["figure.autolayout"] = True

# x = np.array(arr)  # currently the arr vals - > fix to be Date instead
x = df["Date"].values
# x = x[: 0]
y = np.sort(list(arr))

plt.scatter(None, y, color="red")

plt.show()
