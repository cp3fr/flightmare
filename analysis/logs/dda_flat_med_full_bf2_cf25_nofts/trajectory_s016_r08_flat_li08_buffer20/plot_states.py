import pandas as pd
import matplotlib.pyplot as plt
import sys

plt.style.use("ggplot")
plt.rcParams["figure.dpi"] = 100

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plot_colors = colors[:4]

pos_vars = ["position_x [m]", "position_y [m]", "position_z [m]"]
rot_vars = ["rotation_w [quaternion]", "rotation_x [quaternion]", "rotation_y [quaternion]", "rotation_z [quaternion]"]
vel_vars = ["velocity_x [m/s]", "velocity_y [m/s]", "velocity_z [m/s]"]

subplot_labels = ["Position [m]", "Rotation [quaternion]", "Velocity [m/s]"]

recorded_path = sys.argv[1]
reference_path = "original.csv"
df_recorded = pd.read_csv(recorded_path)
df_reference = pd.read_csv(reference_path)
df_reference = df_reference.loc[df_reference["time-since-start [s]"] <= df_recorded["time-since-start [s]"].max()]

fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(16, 8), dpi=100)

df_recorded.plot(x="time-since-start [s]", y=pos_vars, kind="line", legend=False, ax=ax[0], color=plot_colors)
df_recorded.plot(x="time-since-start [s]", y=rot_vars, kind="line", legend=False, ax=ax[1], color=plot_colors)
df_recorded.plot(x="time-since-start [s]", y=vel_vars, kind="line", legend=False, ax=ax[2], color=plot_colors)

df_reference.plot(x="time-since-start [s]", y=pos_vars, kind="line", legend=False, ax=ax[0], style="--", color=plot_colors)
df_reference.plot(x="time-since-start [s]", y=rot_vars, kind="line", legend=False, ax=ax[1], style="--", color=plot_colors)
df_reference.plot(x="time-since-start [s]", y=vel_vars, kind="line", legend=False, ax=ax[2], style="--", color=plot_colors)

for a, lab in zip(ax, subplot_labels):
    a.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    a.set_ylabel(lab)

plt.tight_layout()
plt.show()

