from os.path import join

import numpy as np
from matplotlib import pyplot as plt

from data_storage.data_store import DataStore
from input.definitions import DataKind
from operation.scale_profile import ScaleTimeOfUseProfile, ScaleByQuadraticOptimization, ScaleByLinearEquation
from utility import configuration


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


def visualize_profile_scaling(place_to_visualize="office"):
    # -------------------------------------
    # setup
    # energy consumption in one month divided in tariff time-slots (kWh)
    if place_to_visualize == "office":
        x = np.array([191.4, 73.8, 114.55])
    elif place_to_visualize == "sport centre":
        x = np.array([151.8, 130.3, 127.1])  # sport centre
    else:
        x = np.array([200, 100, 300])

    # number of days of each day-type in the month
    nd = np.array([22, 4, 5])
    # maximum demand allowed
    y_max = 1.25
    # options for methods that scale a reference profile
    #  assigned
    y_ref_db = {"office": np.array(
        [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.3, 0.4, 0.65, 0.9, 0.95, 1, 1, 0.95, 0.9, 0.85, 0.65, 0.45, 0.4, 0.35,
         0.25, 0.25, 0.25, 0.25]), "sport centre": np.array(
        [0.3, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.35, 0.45, 0.55, 0.6, 0.6, 0.6, 0.6, 0.6, 0.7, 0.85, 1, 1, 1,
         0.95, 0.75, 0.55])}
    y_ref = np.repeat(y_ref_db[place_to_visualize][np.newaxis, :], 3, axis=0).flatten()
    # options for 'q_opt' method
    qopt_obj = 0
    qopt_obj_reg = 0
    # ------------------------------------
    # methods
    # method 'scale_gse'
    y_gse, _ = ScaleTimeOfUseProfile()(x, y_ref)
    # method 'scale_seteq'
    y_seteq, stat = ScaleByLinearEquation()(x, nd, y_ref)
    print(f"Scale, set_eq {place_to_visualize}: ", stat)
    # method 'scale_qopt'
    y_qopt, stat = ScaleByQuadraticOptimization(x, nd, y_ref, y_max=y_max, obj=qopt_obj, obj_reg=qopt_obj_reg)
    print(f"Scale, q_opt {place_to_visualize}: ", stat)
    # ------------------------------------
    # plot
    # figure settings
    figsize = (20, 10)
    fontsize = 25
    f_styles = [dict(color='red', alpha=0.1), dict(color='yellow', alpha=0.1), dict(color='green', alpha=0.1)]
    fig, ax = plt.subplots(figsize=figsize)
    number_of_time_steps = configuration.config.getint("time", "number_of_time_steps_per_day")
    dt = 24 / number_of_time_steps
    time_step = 0.5 * dt
    time = np.arange(time_step, number_of_time_steps, time_step)
    # plot profiles
    ax.plot(time, y_gse, label='Scale, gse', marker='s', lw=2, ls='-', )
    ax.plot(time, y_seteq, label='Scale, seteq', marker='s', lw=2, ls='-', )
    ax.plot(time, y_qopt, label='Scale, qopt', marker='s', lw=2, ls='-', )
    # plot reference profile
    if place_to_visualize:
        ax.plot(time, y_ref, color='k', ls='--', lw=1, label='Ref')
    # plot line for 'y_max'
    if y_max:
        ax.axhline(y_max, color='red', ls='--', lw=1, label='y max')
    # plot division into tariff time-slots
    f_sw_pos = []
    f_sw_styles = []
    h0 = 0
    tariff_time_slots = configuration.config.getarray("tariff", "tariff_time_slots", int).flatten()
    for h in range(1, number_of_time_steps + 1):
        previous = tariff_time_slots[h0]
        if h >= number_of_time_steps or tariff_time_slots[h] != previous:
            f_sw_pos.append((h0, h))
            f_sw_styles.append(f_styles[fs.index(previous)])
            h0 = h
    for pos, style in zip(f_sw_pos, f_sw_styles):
        ax.axvspan(*pos, **style, )
    for h in range(0, number_of_time_steps, dt):
        ax.axvline(h, color='grey', ls='-', lw=1)  # ax settings
    ax.set_xlabel("Time (h)", fontsize=fontsize)
    ax.set_ylabel("Power (kW)", fontsize=fontsize)
    ax.set_xticks((np.append(time, time[-1] + dt) - dt / 2)[::4])
    ax.set_xlim([0, time[-1] + dt / 2])
    ax.tick_params(labelsize=fontsize)
    ax.legend(fontsize=fontsize, bbox_to_anchor=(1.005, 0.5), loc="center left")
    ax.grid(axis='y')
    fig.subplots_adjust(top=0.98, bottom=0.1, left=0.1, right=0.8)
    plt.show()
    fig.savefig(join(configuration.config.get("path", "figures"), 'test_methods_scaling.png'), dpi=300)
    plt.close(fig)
