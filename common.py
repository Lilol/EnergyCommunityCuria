# -*- coding: utf-8 -*-
"""
Module 'common.py'
____________
DESCRIPTION
______
NOTES
Notation used in the variables
  - 'f' : tariff timeslot index, \in [1, nf] \subset N
  - 'j' : day-type index, \in [0, n_j) \subset N
  - 'i' : time step index during one day, \in [0, n_i) \subset N
  - 'h' : time step index in multiple days, \in [0, n_h) \subset N
_____
INFO
Author : G. Lorenti (gianmarco.lorenti@polito.it)
Date : 17.11.2022 (last update : 18.11.2022)
"""
# ----------------------------------------------------------------------------
# Libs, packages, modules
import numpy as np
import pandas as pd
# ----------------------------------------------------------------------------
# Common
# months of the year
nm = 12
ms = range(1, nm+1)
months = ['january', 'february', 'march', 'april', 'may', 'june',
          'july', 'august', 'september', 'october', 'november', 'december']
# ARERA's day-types depending on subdivision into tariff time-slots
# NOTE : f : 1 - tariff time-slot F1, central hours of work-days
#            2 - tariff time-slot F2, evening of work-days, and saturdays
#            3 - tariff times-lot F2, night, sundays and holidays
arera = pd.read_csv("Common\\arera.csv", sep=';', index_col=0).values
# total number and list of tariff time-slots (index f)
fs = list(np.unique(arera))
nf = len(fs)
# number of day-types (index j)
# NOTE : j : 0 - work-days (monday-friday)
#            1 - saturdays
#            2 - sundays and holydays
nj = np.size(arera, axis=0)
js = list(range(nj))
# number of time-steps during each day (index i)
ni = np.size(arera, axis=1)
# total number of time-steps (index h)
nh = arera.size
# time-steps where there is a change of tariff time-slot
h_switch_arera = list(
    np.where(np.diff(np.insert(arera.flatten(), -1, arera[0,0])) != 0)[0])
# reference profiles from GSE
y_ref_gse = pd.read_csv("Common\\y_ref_gse.csv", sep=';', index_col=0)
y_ref_gse = {i: row.values
             for i, row in y_ref_gse.set_index(['type', 'month']).iterrows()}

# ----------------------------------------------------------------------------
# Data names and stuff
# Name of the relevant columns in the plants, users and bills files
col_user = 'user'  # code or name of the associated end user
col_municipality = 'municipality'  # municipality
col_description = 'description'  # description of the associated end user
col_type = 'type'  # type of end user
col_address = 'address'  # address of the end user
col_size = 'power'  # size of the plant / available power of the end-user (kW)
col_energy = 'energy'  # annual energy produced / consumed (kWh)
col_yield = 'yield'  # specific annual production (kWh/kWp)
cols_tou_energy = [f'energy_f{f}' for f in range(nf+1)]
col_year = 'year' # year
col_season = 'season'  # season (1-Winter-December to February, ...)
col_month = 'month' # number of the month (1-12)
col_week = 'week'  # number of week of the year
col_day = 'day'  # number of day in the month (1-28, 29 30, 31)
col_daytype = 'day_type'  # type of day (0-work, 1-Saturday, 2-Sunday/holiday)
col_dayweek = 'day_week'  # number of day in the week (1-Monday, ...)

# ----------------------------------------------------------------------------
# Reference year
ref_year = 2019
df_year = pd.read_csv("years_list.csv", sep=';')
labels_ref = []
nds_ref = []
for im, m in enumerate(ms):
    nds_ref.append(
        df_year[((df_year[col_year] == ref_year) &
                 (df_year[col_month] == m))] \
            .groupby(col_daytype).count().iloc[:, 0].values)
    ls = df_year[((df_year[col_year] == ref_year) &
                  (df_year[col_month] == m))][col_daytype].astype(int).values
    labels_ref.append([l+im*nj for l in ls])
labels_ref = np.concatenate(labels_ref)
nds_ref = np.array(nds_ref)