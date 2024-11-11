import numpy as np
from pandas import DataFrame, concat

from utility import configuration
from input.definitions import ColumnName


def create_profiles(user_data, ni, nj, nm, ms):
    output_df = DataFrame()
    for user, df in user_data.groupby(ColumnName.USER):
        # Evaluate profiles in typical days
        months = np.repeat(df.loc[:, ColumnName.MONTH], ni)
        day_types = np.repeat(df.loc[:, ColumnName.DAY_TYPE], ni)
        profiles = df.loc[:, 0:].values.flatten()
        profiles = create_yearly_profile(profiles, months, day_types).reshape((nm, nj * ni))
        # Evaluate typical profiles in each month
        nds = df.groupby([ColumnName.MONTH, ColumnName.DAY_TYPE]).count().iloc[:, 0].values.reshape(nm, nj)
        tou_energy = []
        for y, nd in zip(profiles, nds):
            tou_energy.append(get_monthly_consumption(y, nd))
        # Create dataframe
        tou_energy = np.concatenate((np.full((nm, 1), np.nan), np.array(tou_energy)), axis=1)
        tou_energy = DataFrame(tou_energy, columns=configuration.config.getarray("tariff", "time_of_use_labels", str))
        tou_energy.insert(0, ColumnName.USER, user)
        tou_energy.insert(1, ColumnName.YEAR, configuration.config.getint("time", "year"))
        tou_energy.insert(2, ColumnName.MONTH, ms)
        # Concatenate
        output_df = concat((output_df, tou_energy), axis="rows")
    return output_df
