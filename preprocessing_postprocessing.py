from os.path import join

import pandas as pd

import configuration


if __name__ == '__main__':
    data_users = pd.read_csv("data_users_year.csv", sep=';')
    for user, df in data_users.groupby("user"):
        df.loc[:, "0":].to_csv(join("Loads", f"{user}.csv"), index=False, sep=';')

    data_plants = pd.read_csv("data_plants_year.csv", sep=';')
    for plant, df in data_plants.groupby("user"):
        df.loc[:, "0":].to_csv(join("Generators", f"{plant}.csv"), index=False, sep=';')

    n_fam = configuration.config.getint("rec", "n_families")
    data_fam = pd.read_csv("data_fam_year.csv", sep=';')
    df = data_fam.loc[:, "0":] * n_fam
    df.to_csv(join("Loads", f"famiglie_{n_fam}.csv"), index=False, sep=';')
