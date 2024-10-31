import numpy as np
from matplotlib import pyplot as plt

from input.definitions import ColumnName
from preprocessing import data_users_year, data_users


def vis_profiles(data_fam_year):
    # Families profiles
    # By month
    plt.figure()
    data = data_fam_year.groupby(['user', 'month']).mean().groupby('month').sum().loc[:, 0:]
    for m, profile in data.iterrows():
        plt.plot(profile, label=str(m))
    plt.legend()
    plt.xlabel('Time, h')
    plt.ylabel('Power, kW')
    plt.title('Families profiles')
    plt.show()
    # Whole year
    plt.figure()
    profiles = np.zeros(8760, )
    for _, profile in data_fam_year.groupby(ColumnName.USER):
        profiles += profile.loc[:, 0:].values.flatten()
    plt.plot(profiles, )
    plt.xlabel('Time, h')
    plt.ylabel('Power, kW')
    plt.title('Families profiles')
    plt.show()
    plt.close()


def by_month_profiles(data_plants_year):
    # Production profiles
    # By month
    plt.figure()
    data = data_plants_year.groupby(['user', 'month']).mean().groupby('month').sum().loc[:, 0:]
    for m, profile in data.iterrows():
        plt.plot(profile, label=str(m))
    plt.legend()
    plt.xlabel('Time, h')
    plt.ylabel('Power, kW')
    plt.title('Production profiles')
    plt.show()
    # Whole year
    plt.figure()
    profiles = np.zeros(8760, )
    for _, profile in data_plants_year.groupby(ColumnName.USER):
        profiles += profile.loc[:, 0:].values.flatten()
    plt.plot(profiles, )
    plt.xlabel('Time, h')
    plt.ylabel('Power, kW')
    plt.title('Production profiles')
    plt.show()
    plt.close()


def consumption_profiles():
    # Consumption profiles
    for filter in ['bta', 'ip']:
        data = data_users_year.loc[data_users_year[ColumnName.USER].isin(
            data_users.loc[data_users[ColumnName.USER_TYPE] == filter, ColumnName.USER])]

        # By month
        plt.figure()
        ddata = data.groupby(['user', 'month']).mean().groupby('month').sum().loc[:, 0:]
        for m, profile in ddata.iterrows():
            plt.plot(profile, label=str(m))
        plt.legend()
        plt.xlabel('Time, h')
        plt.ylabel('Power, kW')
        plt.title(f'Consumption profiles {filter.upper()}')
        plt.show()
        # Whole year
        plt.figure()
        profiles = np.zeros(8760, )
        for _, profile in data.groupby(ColumnName.USER):
            profiles += profile.loc[:, 0:].values.flatten()
        plt.plot(profiles, )
        plt.xlabel('Time, h')
        plt.ylabel('Power, kW')
        plt.title(f'Consumption profiles {filter.upper()}')
        plt.show()
        plt.close()

        # Monthly consumption
        plt.figure()
        real = data_users.loc[data_users[ColumnName.USER_TYPE] == filter].set_index([ColumnName.USER]).sort_index()[
            ColumnName.ANNUAL_ENERGY]
        estim = data.groupby('user').sum().sort_index().loc[:, 0:].sum(axis=1)
        plt.barh(range(0, 2 * len(real), 2), real, label='Real')
        plt.barh(range(1, 1 + 2 * len(estim), 2), estim, label='Estimated')
        plt.legend()
        plt.yticks(range(0, 2 * len(real), 2), real.index)
        plt.xlabel('Energy, kWh')
        plt.title(f'Yearly consumption {filter.upper()}')
        plt.show()
        plt.close()
