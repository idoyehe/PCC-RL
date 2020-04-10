import matplotlib.pyplot as plt
import numpy as np

res_output_files = [
    ("res_EpRewMean.txt", 'Episode Cumulative reward'),
    # ("res_loss_ent.txt",),
    ("res_loss_pol_entpen.txt", "Policy Entropy Loss"),
    ("res_loss_vf_loss.txt", "Value Function Loss"),
]

for filename in res_output_files:
    res_file = open(filename[0], "r")
    all_lines = [float(line.split()[1]) for line in res_file.readlines()]

    # Data for plotting
    x_data = np.arange(0, 1024, 1)
    y_data = all_lines

    fig, ax = plt.subplots()
    ax.plot(x_data, y_data)

    ax.set(xlabel='# Iteration', ylabel=filename[1],
           title="{} Vs. Iterations".format(filename[1]))
    ax.grid()

    fig.savefig("{}.jpeg".format(filename[0]))
    plt.show()
