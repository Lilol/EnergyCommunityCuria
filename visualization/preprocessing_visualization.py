from matplotlib import pyplot as plt

from data_storage.data_store import DataStore
from input.definitions import DataKind


def plot_family_profiles(data_fam_year, **kwargs):
    for m, ds in data_fam_year.groupby(DataKind.MUNICIPALITY.value):
        plot_monthly_consumption(ds.squeeze(dim="municipality"), f'Families profiles in municipality {m}')


def plot_pv_profiles(data_plants_year, **kwargs):
    for m, ds in data_plants_year.groupby(DataKind.MUNICIPALITY.value):
        plot_monthly_consumption(ds.squeeze(dim="municipality"), f'Production profiles in municipality {m}')


def plot_consumption_profiles(yearly_consumption_profiles, **kwargs):
    # Consumption profiles
    data_store = DataStore()
    user_data = data_store["users"]
    if yearly_consumption_profiles is None:
        yearly_consumption_profiles = data_store["yearly_load_profiles_from_bills"]
    for m, ds in yearly_consumption_profiles.groupby(DataKind.MUNICIPALITY.value):
        for user_type, real in user_data.groupby(
                user_data.sel({DataKind.USER_DATA.value: DataKind.USER_TYPE}).squeeze()):
            data = ds.sel({DataKind.USER.value: user_data.loc[
                (user_data.sel({DataKind.USER_DATA.value: DataKind.USER_TYPE}) == user_type).squeeze(), m][
                DataKind.USER.value].values}).squeeze()

            # By month
            plot_monthly_consumption(data,
                                     f'Consumption profiles of {user_type.value.upper()} users\nMunicipality: {m}')

            # Monthly consumption
            plt.figure()
            real = real.sortby(real.sel({DataKind.USER_DATA.value: DataKind.ANNUAL_ENERGY}).squeeze()).sel(
                {DataKind.USER_DATA.value: DataKind.ANNUAL_ENERGY}).squeeze().sortby(DataKind.USER.value)
            estim = data.groupby(DataKind.USER.value).sum(dim=DataKind.TIME.value).sortby(DataKind.USER.value)
            plt.barh(range(0, 2 * len(real), 2), real, label='Real')
            plt.barh(range(1, 1 + 2 * len(estim), 2), estim, label='Estimated')
            plt.legend()
            plt.yticks(range(0, 2 * len(real), 2), real[DataKind.USER.value].values)
            plt.xlabel('Energy, kWh')
            plt.title(f'Yearly consumption of {user_type.value.upper()} users in {m}')
            plt.tight_layout()
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
    plt.tight_layout()
    plt.show()
    plt.close()

    # Whole year
    plt.figure()
    data.sum(dim=DataKind.USER.value).to_pandas().plot()
    plt.xlabel('Time, h')
    plt.ylabel('Power, kW')
    plt.title(title)
    plt.tight_layout()
    plt.show()
    plt.close()
