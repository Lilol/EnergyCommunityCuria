# -*- coding: utf-8 -*-
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
# ----------------------------------------------------------------------------
# Import
# python libs, packages, modules
import numpy as np
import pandas as pd
from pathlib import Path
# ----------------------------------------------------------------------------
user_types = dict(
    PDMF = 'dom',
    PAUF = 'bta',
    PICM = 'ip',
    )
basepath = Path(__file__).parent
fname_out = 'gse_ref_profiles.csv'
#
fname_in = 'gse_ref_profiles.xlsx'
df_gse = pd.read_excel(basepath/fname_in)
#
fname_in = 'years_list.xlsx'
df_years = pd.read_excel(basepath/fname_in, sheet_name='years_list')
#
df_gse = df_gse.merge(df_years[['year', 'month', 'day', 'day_type']],
                      on=['year', 'month', 'day'], how='left')
#
df_out = pd.DataFrame()
for user_type in user_types:
    df_gse_user = df_gse[['month','day','hour','day_type', user_type]]
    yref_user = []
    #
    for month in df_gse_user['month'].unique():
        df_gse_groupd = \
            df_gse_user.loc[df_gse_user['month']==month].groupby(
                ['day_type','hour'])
        yref_user.append(df_gse_groupd.mean()[user_type].values)
    df_yref_user = pd.DataFrame(yref_user,
                                columns=['y_j{}_i{}'.format(j, i).replace(' ', '0')
                                         for j in range(3) for i in range(24)])
    df_yref_user.insert(0, 'month', df_gse_user['month'].unique())
    df_yref_user.insert(0, 'type', [user_types[user_type]]*len(df_yref_user))

    df_out = pd.concat((df_out, df_yref_user), axis=0)

df_out.to_csv(fname_out, sep=';')
