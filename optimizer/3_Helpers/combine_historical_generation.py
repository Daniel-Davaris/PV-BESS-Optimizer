import pandas as pd
from pathlib import Path

folder = Path("optimizer/1_Dataset/1_Raw_data/Raw_PV")

data = pd.read_csv(list(folder.glob("*.csv"))[0], encoding='latin-1',skiprows=11,sep=';').iloc[:,[0,4]]

df = []
for i in folder.glob("*.csv"):
    data = pd.read_csv(i, encoding='latin-1',skiprows=11,sep=';').iloc[:,[0,3]]
    df.append(data)

df = pd.concat(df, ignore_index=True)
df.columns=['Date','kW']
df['Date'] = pd.to_datetime(df['Date'],format="%d/%m/%y %H:%M")
df = df.set_index('Date')
df['kW'] = df['kW']/1000

print(df.index.year.value_counts().sort_index())
df.to_csv("optimizer/1_Dataset/1_Raw_data/Historical_generation.csv", date_format="%d/%m/%Y %H:%M")
  