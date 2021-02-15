try:
    from analysis.functions.visattention import *
except:
    from functions.visattention import *

# Plot the reference trajectory.
filepath = './logs/resnet_test/trajectory_reference_original.csv'
ref = trajectory_from_flightmare(filepath=filepath)
ref = ref.iloc[np.arange(0, ref.shape[0], 100), :]
ax = plot_trajectory(ref.px.values, ref.py.values, ref.pz.values,
                     ref.qx.values, ref.qy.values, ref.qz.values, ref.qw.values, c='r')

# # Plot the MPC trajectory.
# filepath = './logs/dda_0/trajectory_mpc_eval_nw.csv'
# mpc = trajectory_from_flightmare(filepath=filepath)
# plot_trajectory(mpc.px, mpc.py, mpc.pz, c='b', ax=ax)
#
# # Plot all trajectories flown by the network
# for w in os.walk('./logs/dda_0/'):
#     for f in w[2]:
#         if (f.find('.csv') != -1) and (f.find('mpc_eval_nw') == -1):
#             filepath = os.path.join(w[0], f)
#             df = trajectory_from_flightmare(filepath=filepath)
#             plot_trajectory(df.px, df.py, df.pz, c='k', ax=ax)

# Format the output figure
ax = format_trajectory_figure(
    ax, xlims=(-30, 30), ylims=(-30, 30), zlims=(-30, 30), xlabel='px [m]',
    ylabel='py [m]', zlabel='pz [m]')

plt.show()

