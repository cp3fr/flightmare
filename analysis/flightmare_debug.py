import os
import sys

# Add base path
base_path = os.getcwd().split('/flightmare/')[0]+'/flightmare/'
sys.path.insert(0, base_path)

from analysis.utils import *
from pathlib import Path

base_path = Path(base_path)

to_check_gates = True
to_check_trajectory = True


debug_path = Path('/media/cp3fr/Elements/rss21_visual_attention/flightmare/analysis/flightmare_debug')
process_path = Path('/media/cp3fr/Elements/rss21_visual_attention/flightmare/analysis/process/dda_flat_med_full_bf2_cf25_noref_nofts_attbr/s016_r05_flat_li01_buffer20/mpc2nw_mt-300_st-20_00')


d1 = pd.read_csv(debug_path/'mpc2nw_mt-300_st-20_00.csv')
w1 = pd.read_csv(debug_path/'flat.csv')
w1 = w1.rename(columns={
    'pos_x': 'px',
    'pos_y': 'py',
    'pos_z': 'pz',
    'rot_x_quat': 'qx',
    'rot_y_quat': 'qy',
    'rot_z_quat': 'qz',
    'rot_w_quat': 'qw',
    'dim_x': 'dx',
    'dim_y': 'dy',
    'dim_z': 'dz',
    })
w1['dx'] = 0
w1['dy'] = 3
w1['dz'] = 3


d2 = pd.read_csv(process_path/'trajectory.csv')
w2 = pd.read_csv(process_path/'track.csv')


fig = plt.figure()
ax = fig.add_subplot(1,1,1,projection='3d')


if to_check_gates:
    ax=plot_gates_3d(track=w1, ax=ax, color='b')
    ax=plot_gates_3d(track=w2, ax=ax, color='r')


if to_check_trajectory:
    ax.plot(d1['position_x [m]'].values,
             d1['position_y [m]'].values,
             d1['position_z [m]'].values,
             )
    ax.plot(d2['px'].values,
             d2['py'].values,
             d2['pz'].values,
             )



ax.set_xlim((-20, 20))
ax.set_ylim((-20, 20))
ax.set_zlim((-20, 20))


plt.show()
