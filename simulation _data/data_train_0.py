import pandas as pd

url = "https://raw.githubusercontent.com/AnshTanwar/credit-ease/main/server/Training%20Data.csv"
data = pd.read_csv(url)
# save data to csv
data.to_csv("credit_data.csv", index=False)
print(data.head())
