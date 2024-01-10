import os
import pandas as pd
path = "C:/Users/Raul/OneDrive - UAB/4t/MA PSIV/RETO EPILIEPSIA/metadata"

data = None

parquet_files = os.listdir(path)
for parquet in parquet_files:
    df = pd.read_parquet(os.path.join(path, parquet))
    df['window_id'] = df.index
    df['filename'] = df['filename'].apply(lambda x: x.split("_")[0])
    if data is None:
        data = df
    else:
        data = pd.concat([data, df])

print(len(data))

for idx in range(len(data)):
    row = data.iloc[idx]
    id, window_id, cls = row['filename'], row['window_id'], row['class']
    if id == "chb17a" or window_id == "chb17a":
        print("ENCONTRADO")
        print(row)
        break