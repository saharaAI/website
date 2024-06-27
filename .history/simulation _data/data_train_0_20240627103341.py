import pandas as pd

url = "https://raw.githubusercontent.com/AnshTanwar/credit-ease/main/server/Training%20Data.csv"
data = pd.read_csv(url)

print(data.head())
