# -*- coding: utf-8 -*-

# TODO: What to do with this file?

"""
Module 'gse_ref_profiles.py'
____________
DESCRIPTION
______
NOTES
_____
INFO
Author : G. Lorenti (gianmarco.lorenti@polito.it)
Date : 22.11.2022 (last update : dd.mm.yyyy)
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Import
# python libs, packages, modules
import pandas as pd

from input.definitions import UserType, DataKind
from time.day_of_the_week import df_year

# ----------------------------------------------------------------------------
basepath = Path(__file__).parent
fname_out = 'gse_ref_profiles.csv'

#
fname_in = 'gse_ref_profiles.xlsx'
df_gse = pd.read_excel(basepath / fname_in)

#
df_gse = df_gse.merge(df_year[[DataKind.YEAR, DataKind.MONTH, DataKind.DAY_OF_MONTH, DataKind.DAY_TYPE]],
                      on=[DataKind.YEAR, DataKind.MONTH, DataKind.DAY_OF_MONTH], how='left')
#
df_out = pd.DataFrame()
for user_type in UserType:
    df_gse_user = df_gse[
        [DataKind.MONTH, DataKind.DAY_OF_MONTH, DataKind.HOUR, DataKind.DAY_TYPE, user_type.value]]
    yref_user = []
    #
    for month in df_gse_user[DataKind.MONTH].unique():
        df_gse_groupd = df_gse_user.loc[df_gse_user[DataKind.MONTH] == month].groupby(
            [DataKind.DAY_TYPE, DataKind.HOUR])
        yref_user.append(df_gse_groupd.mean()[user_type].values)
    df_yref_user = pd.DataFrame(yref_user,
                                columns=[f'y_j{j}_i{i}'.replace(' ', '0') for j in range(3) for i in range(24)])
    df_yref_user.insert(0, DataKind.MONTH, df_gse_user[DataKind.MONTH].unique())
    df_yref_user.insert(0, DataKind.USER_TYPE, [user_type.value, ] * len(df_yref_user))
    df_out = pd.concat((df_out, df_yref_user), axis=0)

df_out.to_csv(fname_out, sep=';')
