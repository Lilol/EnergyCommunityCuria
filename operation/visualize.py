import matplotlib.pyplot as plt
import numpy as np

from operation.scale_profile import ScaleTimeOfUseProfile


def visualize_profile_scaling():
    # -------------------------------------
    # setup
    # energy consumption in one month divided in tariff time-slots (kWh)
    x = np.array([151.8, 130.3, 127.1])  # sport centre
    x = np.array([191.4, 73.8, 114.55])  # office
    x = np.array([200, 100, 300])
    # number of days of each day-type in the month
    nd = np.array([22, 4, 5])
    # maximum demand allowed
    y_max = 1.25
    # time-step on which to "anchor" load profiles in each day-type
    i_anch = 0
    # options for methods that scale a reference profile
    #  assigned
    ref = 'office'
    y_ref_db = dict(office=np.array(
        [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.3, 0.4, 0.65, 0.9, 0.95, 1, 1, 0.95, 0.9, 0.85, 0.65, 0.45, 0.4, 0.35,
         0.25, 0.25, 0.25, 0.25]), sport_centre=np.array(
        [0.3, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.35, 0.45, 0.55, 0.6, 0.6, 0.6, 0.6, 0.6, 0.7, 0.85, 1, 1, 1,
         0.95, 0.75, 0.55]), )
    y_ref = np.repeat(y_ref_db[ref][np.newaxis, :], 3, axis=0)
    y_ref = y_ref.flatten()
    # options for 'q_opt' method
    qopt_obj = 0
    qopt_obj_reg = 0
    # ------------------------------------
    # methods
    # method 'scale_gse'
    y_gse, _ = ScaleTimeOfUseProfile()(x, y_ref)
    # print ("Scale, set_eq {}: ".format(ref), stat)
    # method 'scale_seteq'
    y_seteq, stat = scale_seteq(x, nd, y_ref)
    print(f"Scale, set_eq {ref}: ", stat)
    # method 'scale_qopt'
    y_qopt, stat = scale_qopt(x, nd, y_ref, y_max=y_max, obj=qopt_obj, obj_reg=qopt_obj_reg)
    print(f"Scale, q_opt {ref}: ", stat)
    # ------------------------------------
    # plot
    # figure settings
    figsize = (20, 10)
    fontsize = 25
    f_styles = [dict(color='red', alpha=0.1), dict(color='yellow', alpha=0.1), dict(color='green', alpha=0.1)]
    fig, ax = plt.subplots(figsize=figsize)
    time = np.arange(nh) + 0.5 * (dt := 24 / ni)
    # plot profiles
    ax.plot(time, y_gse, label='Scale, gse', marker='s', lw=2, ls='-', )
    ax.plot(time, y_seteq, label='Scale, seteq', marker='s', lw=2, ls='-', )
    ax.plot(time, y_qopt, label='Scale, qopt', marker='s', lw=2, ls='-', )
    # plot reference profile
    if ref:
        ax.plot(time, y_ref, color='k', ls='--', lw=1, label='Ref')
    # plot line for 'y_max'
    if y_max:
        ax.axhline(y_max, color='red', ls='--', lw=1, label='y max')
    # plot division into tariff time-slots
    f_sw_pos = []
    f_sw_styles = []
    h0 = 0
    for h in range(1, nh + 1):
        if h >= nh or arera.flatten()[h] != arera.flatten()[h0]:
            f_sw_pos.append((h0, h))
            f_sw_styles.append(f_styles[fs.index(arera.flatten()[h0])])
            h0 = h
    for pos, style in zip(f_sw_pos, f_sw_styles):
        ax.axvspan(*pos, **style, )
    for h in range(0, nh, ni):
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
    plt.close(fig)  # fig.savefig('test_methods_scaling.png', dpi=300)