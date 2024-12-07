import numpy as np
from matplotlib import pyplot as plt


def plot_shared_energy(sh1, sh2, n_fam):
    plt.figure()
    plt.plot(np.diff(sorted(sh2 - sh1)), label=f"{n_fam}", color='lightgrey')
    plt.yticks([])
    plt.xlabel('Numero giorni dell\' anno')
    plt.twinx().plot(sorted(sh2 - sh1), label=f"{n_fam}")
    plt.ylabel('Gap tra energia condivisa oraria e giornaliera (kWh)')
    plt.gca().yaxis.set_label_position("left")
    plt.gca().yaxis.tick_left()
    plt.title(f"Numero famiglie: {int(n_fam)}")
    plt.show()
    plt.close()


def plot_sci(time_resolution, n_fams, results):
    plt.figure()
    for label in time_resolution:
        plt.plot(n_fams, results[label], label=label)
    plt.plot(n_fams, results['sc_tou'], label='sc_tou', color='lightgrey', ls='--')
    # plt.scatter(n_fams, scs, label='evaluated')
    plt.xlabel(f'Numero famiglie: {n_fams}')
    plt.ylabel('SCI')
    plt.legend()
    plt.show()
    plt.close()
