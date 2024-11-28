from matplotlib import pyplot as plt

from data_storage.data_store import DataStore
from input.definitions import DataKind, UserType


def plot_family_profiles(data_fam_year):
    plot_monthly_consumption(data_fam_year, 'Families profiles')


def plot_pv_profiles(data_plants_year):
    plot_monthly_consumption(data_plants_year, 'Production profiles')


def plot_consumption_profiles(yearly_consumption_profiles):
    # Consumption profiles
    user_data = DataStore()["users"]
    for user_type in (UserType.PAUF, UserType.PICM):
        data = yearly_consumption_profiles.sel({DataKind.USER.value: user_data.loc[
            (user_data.sel({DataKind.USER_DATA.value: DataKind.USER_TYPE}) == user_type).squeeze()][
            DataKind.USER.value]})

        # By month
        plot_monthly_consumption(data, f'Consumption profiles of {user_type.value.upper()} users')

        # Monthly consumption
        plt.figure()
        real = user_data.loc[user_data[DataKind.USER_TYPE] == user_type].set_index([DataKind.USER]).sort_index()[
            DataKind.ANNUAL_ENERGY]
        estim = data.groupby('user').sum().sort_index().loc[:, 0:].sum(axis=1)
        plt.barh(range(0, 2 * len(real), 2), real, label='Real')
        plt.barh(range(1, 1 + 2 * len(estim), 2), estim, label='Estimated')
        plt.legend()
        plt.yticks(range(0, 2 * len(real), 2), real.index)
        plt.xlabel('Energy, kWh')
        plt.title(f'Yearly consumption of {user_type.value.upper()}')
        plt.show()
        plt.close()


def plot_monthly_consumption(data, title):
    # Aggregated by month
    plt.figure()
    data.groupby(data.time.dt.month).mean().T.to_pandas().plot()
    plt.legend()
    plt.xlabel('Time, h')
    plt.ylabel('Power, kW')
    plt.title(title)
    plt.show()

    # Whole year
    plt.figure()
    data.sum(dim=DataKind.USER.value).to_pandas().plot()
    plt.xlabel('Time, h')
    plt.ylabel('Power, kW')
    plt.title(title)
    plt.show()
    plt.close()
