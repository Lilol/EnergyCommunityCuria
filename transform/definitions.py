# TODO: option for data processing methodology (will be more in the future)
import numpy as np
from pandas import DataFrame, concat

from input.definitions import ColumnName
from input.reader import BillsReader
from preprocessing import year
from transform.utils import eval_y_from_year, eval_x


def create_profiles(user_data, ni, nj, nm, ms):
    output_df = DataFrame()
    for user, df in user_data.groupby(ColumnName.USER):
        # Evaluate profiles in typical days
        months = np.repeat(df.loc[:, ColumnName.MONTH], ni)
        day_types = np.repeat(df.loc[:, ColumnName.DAY_TYPE], ni)
        profiles = df.loc[:, 0:].values.flatten()
        profiles = eval_y_from_year(profiles, months, day_types).reshape((nm, nj * ni))
        # Evaluate typical profiles in each month
        nds = df.groupby([ColumnName.MONTH, ColumnName.DAY_TYPE]).count().iloc[:, 0].values.reshape(nm, nj)
        tou_energy = []
        for y, nd in zip(profiles, nds):
            tou_energy.append(eval_x(y, nd))
        # Create dataframe
        tou_energy = np.concatenate((np.full((nm, 1), np.nan), np.array(tou_energy)), axis=1)
        tou_energy = DataFrame(tou_energy, columns=BillsReader.time_of_use_energy_column_names)
        tou_energy.insert(0, ColumnName.USER, user)
        tou_energy.insert(1, ColumnName.YEAR, year)
        tou_energy.insert(2, ColumnName.MONTH, ms)
        # Concatenate
        output_df = concat((output_df, tou_energy), axis=0)
    return output_df
