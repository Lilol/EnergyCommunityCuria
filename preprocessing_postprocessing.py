import pandas as pd
from pathlib import Path
import os

data_users = pd.read_csv("data_users_year.csv", sep=';')
for user, df in data_users.groupby("user"):
    df.loc[:, "0":].to_csv(f"Loads/{user}.csv", index=False, sep=';')

data_plants = pd.read_csv("data_plants_year.csv", sep=';')
for plant, df in data_plants.groupby("user"):
    df.loc[:, "0":].to_csv(f"Generators/{plant}.csv", index=False, sep=';')

n_fam = 50
data_fam = pd.read_csv("data_fam_year.csv", sep=';')
df = data_fam.loc[:, "0":] * n_fam
df.to_csv(f"Loads/famiglie_{n_fam}.csv", index=False, sep=';')
