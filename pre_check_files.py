# -*- coding: utf-8 -*-
"""
Module '.py'
____________
DESCRIPTION
______
NOTES
_____
INFO
Author : G. Lorenti (gianmarco.lorenti@polito.it)
Date : dd.mm.yyyy (last update : dd.mm.yyyy)
"""
from os.path import join
from pathlib import Path
from string import ascii_uppercase as alphabet

# ----------------------------------------------------------------------------
# Import
# libs, packages, modules
import numpy as np
import openpyxl
import pandas as pd
from openpyxl.styles import PatternFill

from utility import configuration

# ----------------------------------------------------------------------------
# ERROR LEGEND
# POD file -----------------------
# yellow: missing a required value
# red: invalid value
# blue POD: POD without bill
# Bills file ---------------------
# yellow: missing a required value
# red: invalid value
# blue POD: POD without POD
# orange year: year with more or less than 12 monthly records
# orange month: month with more than 3 records
# yellow consumption: missing or 'nd' values
# red consumption: negative or other invalid values
# orange F0 if the same user has both F0 and F1, F2, F3
# ----------------------------------------------------------------------------
# Setup

# input folder (path relative to this file or full path)

input_folder = join(configuration.config.get("path", "rec_data"), "TestCesana")
input_path = Path(input_folder)
# names of the two input files
fname_pods = "lista_pod.xlsx"
fname_bills = "dati_bollette.xlsx"
# output folder
output_folder = ""
output_path = Path(output_folder)
# names of the files in output
fname_pods_out = "lista_pod_check.xlsx"
fname_bills_out = "dati_bollette_check.xlsx"

# name of the column with pod, common to both DataFrames
c_pod = 'pod'
# name of the column with month, in bills DataFrame
c_month = 'mese'
# name of the column with year, in bills DataFrame
c_year = 'anno'
# name of the column with F0, in bills DataFrame
c_f0 = 'f0'
# name of the column with F1, in bills DataFrame
c_f1 = 'f1'
# name of the column with F2, in bills DataFrame
c_f2 = 'f2'
# name of the column with F3, in bills DataFrame
c_f3 = 'f3'

# automatic check for PODs file (column, format, check)
check_pods = [(c_pod, 'y', {'check': 'are_notnull'}), ('tipo', 'y', {'check': 'are_notnull'}),
    ('potenza', 'y', {'check': 'are_notnull'}), ('descrizione', 'y', {'check': 'are_notnull'}),
    ('tipo', 'r', {'check': 'are_in', 'allowed': ['ip', 'bta', 'mta', 'altro', np.nan]}),
    ('potenza', 'r', {'check': 'are_pos'})]
# automatic checks for bills file (column, format, check)
check_bills = [(c_pod, 'y', {'check': 'are_notnull'}), (c_year, 'y', {'check': 'are_notnull'}),
    (c_month, 'y', {'check': 'are_notnull'}), (c_year, 'r', {'check': 'are_in', 'allowed': range(2018, 2023)}),
    (c_month, 'r', {'check': 'are_in', 'allowed': range(1, 13)}), ]


# ----------------------------------------------------------------------------
# Useful functions

# test a check on a series
def test_check(series, check, **kwargs):
    assert isinstance(series, pd.Series), f"Can only check a series object, not {type(series)}."

    if check == 'are_notnull':
        return ~series.isnull()

    if check == 'are_in':
        assert 'allowed' in kwargs, "Missing allowed values for 'are_in' type check."
        allowed = kwargs['allowed']
        return series.isin(allowed)

    if check == 'are_pos':
        return series > 0

    if check == 'are_notneg':
        return series >= 0

    raise ValueError(f"Invalid check '{check}.")


# ----------------------------------------------------------------------------
# Data checking
# --------------------------------------
# data loading

# file with PODs list
df_pods = pd.read_excel(input_path / fname_pods)
# add required columns
cols_required = set([col for col, _, _ in check_pods])
cols_to_add = [col for col in cols_required if col not in df_pods.columns]
df_pods[cols_to_add] = np.nan
# add column to assess if rows are okay
df_pods[c_test := 'test'] = True

# file with bills list
df_bills = pd.read_excel(input_path / fname_bills)
# add required columns
cols_required = set([col for col, _, _ in check_bills])
cols_to_add = [col for col in cols_required if col not in df_bills.columns]
df_bills[cols_to_add] = np.nan
# add column to assess if rows are okay
df_bills['test'] = True

# --------------------------------------
# initialise stuff

# dict of cell to format as "yellow" in PODs Excel file (cell:format)
cells_fmt_pods = {}
# dict of cell to format as "yellow" in bills Excel file (cell:format)
cells_fmt_bills = {}

# --------------------------------------
# add combined check for pods in POD list and bills list
check_pods.append((c_pod, 'b', {'check': 'are_in', 'allowed': df_bills[c_pod].unique()}), )

check_bills.append((c_pod, 'b', {'check': 'are_in', 'allowed': df_pods[c_pod].unique()}), )

# --------------------------------------
# check PODs file
for col, fmt, check in check_pods:
    # make sure 'col' is in DataFrame
    assert col in df_pods, f"Column to check not in DataFrame: {col}."
    # test
    test = test_check(df_pods[col], **check)
    # select rows that do not pass the test
    rows = list(test.loc[test == False].index)
    # update cells to be formatted
    cells_fmt_pods.update({(col, row): fmt for row in rows})
    # update test column
    df_pods.loc[rows, c_test] = False

# --------------------------------------
# check bills file
for col, fmt, check in check_bills:
    # make sure 'col' is in DataFrame
    assert col in df_bills, f"Column to check not in DataFrame: {col}."
    # test
    test = test_check(df_bills[col], **check)
    # select rows that do not pass the test
    rows = list(test.loc[test == False].index)
    # update cells to be formatted
    cells_fmt_bills.update({(col, row): fmt for row in rows})
    # update test column
    df_bills.loc[rows, c_test] = False

# ---------------------------------------
# EXTRA checks on bills file
# check that each month has exactly one record
for group, dfg in df_bills.groupby([c_pod, c_year, c_month]):
    if len(dfg) > 1:
        rows = list(dfg.index)
        cells_fmt_bills.update({(c_month, row): 'o' for row in rows})
        df_bills.loc[rows, c_test] = False
# check that each year has one record for each month
for group, dfg in df_bills.groupby([c_pod, c_year]):
    if len(dfg['anno']) != 12:
        rows = list(dfg.index)
        cells_fmt_bills.update({(c_year, row): 'o' for row in rows})
        df_bills.loc[rows, c_test] = False
# check for negative, empty or 'nd' values in the consumption
for group, dfg in df_bills.groupby([c_pod, c_year]):
    # if f1, f2, f3 are always null
    if dfg[[c_f1, c_f2, c_f3]].isnull().all().all():
        # check for negative or non-numerical values
        test = dfg[c_f0].apply(pd.to_numeric, errors='coerce') >= 0
        if not test.all():
            rows = list(test.loc[test == False].index)
            cells_fmt_bills.update({(c_f0, row): 'r' for row in rows})
            df_bills.loc[rows, c_test] = False
        # check for empty or 'nd' values
        test = ~dfg[c_f0].replace('nd', np.nan).isnull()
        if not test.all():
            rows = list(test.loc[test == False].index)
            cells_fmt_bills.update({(c_f0, row): 'y' for row in rows})
            df_bills.loc[rows, c_test] = False
    # otherwise check only f1, f2, f3
    else:
        # check for negative or non-numerical values
        test = dfg[[c_f1, c_f2, c_f3]].apply(pd.to_numeric, errors='coerce') >= 0
        if not test.all().all():
            for col in test:
                rows = list(test.loc[test[col] == False, col].index)
                cells_fmt_bills.update({(col, row): 'r' for row in rows})
                df_bills.loc[rows, c_test] = False
        # check for empty or 'nd' values
        test = ~dfg[[c_f1, c_f2, c_f3]].replace('nd', np.nan).isnull()
        if not test.all().all():
            for col in test:
                rows = list(test.loc[test[col] == False, col].index)
                cells_fmt_bills.update({(col, row): 'y' for row in rows})
                df_bills.loc[rows, c_test] = False
        # however, highlight it anyway since both f0 and f1, f2, f3 are present
        if ~dfg[c_f0].isnull().all():
            rows = list(dfg.index)
            cells_fmt_bills.update({(c_f0, row): 'o' for row in rows})
            df_bills.loc[rows, c_test] = False

# -----------------------------------------------------------------------------
# Save formatted files

# ---------------------------------------
# formats
fmts = {'b': PatternFill(patternType='solid', fgColor='0000FF'),
    'w': PatternFill(patternType='solid', fgColor='FFFFFF'), 'y': PatternFill(patternType='solid', fgColor='FFFF00'),
    'r': PatternFill(patternType='solid', fgColor='FF0000'), 'o': PatternFill(patternType='solid', fgColor='FFA500'), }

# ---------------------------------------
# pods file
# map of df_pods columns and rows into Excel file columns
if len(df_pods.columns) > len(alphabet):
    raise ValueError("Too many columns in PODs file for alphabet")
cols = list(alphabet)[:len(df_pods.columns) % len(alphabet)]
cols_dict_pods = {col_pod: col for col_pod, col in zip(df_pods.columns, cols)}
rows_dict_pods = {i: row for i, row in zip(df_pods.index, range(2, len(df_pods) + 2))}
# remap cells to format
cells_fmt_pods = {f'{cols_dict_pods[col]}{rows_dict_pods[row]}': fmt for (col, row), fmt in cells_fmt_pods.items()}

# save non-formatted file
df_pods.to_excel(output_path / fname_pods_out, sheet_name='dati', index=False)

# re-open to format
wb = openpyxl.load_workbook(output_path / fname_pods_out)
ws = wb['dati']  # Name of the working sheet

# format
for cell, fmt in cells_fmt_pods.items():
    ws[cell].fill = fmts[fmt]

# save formatted
wb.save(output_path / fname_pods_out)

# ---------------------------------------
# bills file

# map of df_bills columns and rows into Excel file columns
if len(df_bills.columns) > len(alphabet):
    raise ValueError("Too many columns in bills file for alphabet")
cols = list(alphabet)[:len(df_bills.columns) % len(alphabet)]
cols_dict_bills = {col_bill: col for col_bill, col in zip(df_bills.columns, cols)}
rows_dict_bills = {i: row for i, row in zip(df_bills.index, range(2, len(df_bills) + 2))}
# remap cells to format
cells_fmt_bills = {f'{cols_dict_bills[col]}{rows_dict_bills[row]}': fmt for (col, row), fmt in cells_fmt_bills.items()}

# save non-formatted file
df_bills.to_excel(output_path / fname_bills_out, sheet_name='dati', index=False)

# re-open to format
wb = openpyxl.load_workbook(output_path / fname_bills_out)
ws = wb['dati']  # Name of the working sheet

# format
for cell, fmt in cells_fmt_bills.items():
    ws[cell].fill = fmts[fmt]

# save formatted
wb.save(output_path / fname_bills_out)
