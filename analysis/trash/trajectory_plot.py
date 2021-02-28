try:
    from analysis.utils import *
except:
    from functions.visattention import *

"""
Comparison of 3D flight trajectories and paths between reference, MPC, 
and network.

Also shows the 3D poses for reference and MPC.
"""

# Make a 3D trajectory plots with poses
filepaths =['./logs/resnet_test/trajectory_reference_original.csv',
            './logs/dda_0/trajectory_mpc_eval_nw.csv']
for filepath in filepaths:
    df = trajectory_from_logfile(filepath=filepath)
    #downsample the data to 20 Hz
    sr = 1 / np.nanmedian(np.diff(df.t.values))
    df = df.iloc[np.arange(0, df.shape[0], int(sr / 20)), :]
    ax = plot_trajectory(
        df.px.values, df.py.values, df.pz.values, df.qx.values,
        df.qy.values, df.qz.values, df.qw.values, axis_length=3, c='k')
    ax = format_trajectory_figure(
        ax, xlims=(-30, 0), ylims=(-15, 15), zlims=(-15, 15), xlabel='px [m]',
        ylabel='py [m]', zlabel='pz [m]', title=filepath)

# Plot reference, MPC, and network trajectories in 3D
filepath = '../logs/resnet_test/trajectory_reference_original.csv'
ref = trajectory_from_logfile(filepath=filepath)
ref = ref.iloc[np.arange(0, ref.shape[0], 50), :]
ax = plot_trajectory(ref.px.values, ref.py.values, ref.pz.values, c='k')
basepath = './logs/dda_0/'
for w in os.walk(basepath):
    for f in w[2]:
        if (f.find('.csv') != -1) :
            if f.find('mpc_eval_nw') != -1:
                color = 'r'
            else:
                color = 'b'
            filepath = os.path.join(w[0], f)
            df = trajectory_from_logfile(filepath=filepath)
            plot_trajectory(df.px, df.py, df.pz, c=color, ax=ax)
ax = format_trajectory_figure(
    ax, xlims=(-30, 30), ylims=(-30, 30), zlims=(-30, 30), xlabel='px [m]',
    ylabel='py [m]', zlabel='pz [m]', title=basepath)

plt.show()

