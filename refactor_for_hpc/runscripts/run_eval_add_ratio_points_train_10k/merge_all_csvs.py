import pandas as pd 
from pathlib import Path 

csv_files = Path(".").rglob("*.csv")

dfs = [] 

for csv_file in csv_files:
    dfs.append(pd.read_csv(csv_file))

dfs = pd.concat(dfs, ignore_index=True)
dfs = dfs.reset_index(drop=True)
dfs.to_csv("all_results.csv")
