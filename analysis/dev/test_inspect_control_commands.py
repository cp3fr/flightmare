import os
import sys

# Add base path
base_path = os.getcwd().split('/flightmare/')[0]+'/flightmare/'
sys.path.insert(0, base_path)

from analysis.utils import *


data_path = './process/dda_flat_med_full_bf2_cf25_noref_nofts_attbr' \
            '/s016_r05_flat_li01_buffer20/mpc_nw_act_00/'

trajectory = pd.read_csv(data_path+'trajectory.csv')
track = pd.read_csv(data_path+'track.csv')



iax = 0
for n1 in ['throttle', 'roll', 'pitch', 'yaw']:
    iax+=1
    plt.subplot(4, 1, iax)
    for n2 in ['mpc', 'nw']:
        plt.plot(trajectory.t, trajectory['{}_{}'.format(n1, n2)])

    plt.ylabel(n1)
    plt.legend(['mpc', 'nw'], loc="upper left")
    if iax==1:
        plt.title(data_path)

plt.show()



ax = plot_trajectory_with_gates_3d(
    trajectory=trajectory,
    track=track,
)

plt.show()