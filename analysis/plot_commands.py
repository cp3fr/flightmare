import os
import sys

from pathlib import Path

import pandas as pd

base_path = Path(os.getcwd().split('/flightmare/')[0]+'/flightmare/')
sys.path.insert(0, base_path.as_posix())

from analysis.utils import *
path={}
path['proc']=base_path/'analysis'/'process'

filepaths = sorted(path['proc'].rglob('*reference.csv'))

for f in filepaths:
    print(f)

    # Read visatt reference trajectory.
    r=pd.read_csv(f)

    # Read visatt trajectory flown by the network.
    t=pd.read_csv(f.parent/'trajectory.csv')

    # Load gazesim drone trajectory.
    datapath='/media/cp3fr/Elements1/GazeSim/data/process'
    track=f.parts[-3].split('-')[1] #track name as in the subfolder name "05_flat"
    subject=int(f.parts[-3].split('-')[2].strip('s')) #subject index as in the folder name "s004"
    run=int(f.parts[-3].split('-')[3].split('_')[1].strip('r')) #run index as in the subfolder number "05_flat"
    lap=int(f.parts[-3].split('-')[3].split('_')[2].strip('li')) #lap index as in laptimes.csv
    buffer=float(f.parts[-3].split('-')[3].split('_')[3].strip('buffer'))/10 #time in sec to add before the start of the lap
    def fix_quaternion(q, initial_value=1):
        """Fix jumps in the quaternion across time by self-similarity check wrt previous sample."""
        qp = np.vstack((q.copy()[:1, :], q.copy()[:-1, :]))
        qn = -np.vstack((q.copy()[:1, :], q.copy()[:-1, :]))
        errp = np.nanmean((qp - q) ** 2, axis=1)
        errn = np.nanmean((qn - q) ** 2, axis=1)
        m = []
        v = initial_value
        for i in range(errp.shape[0]):
            if errp[i] > errn[i]:
                v = -v
            m.append(v)
        m = np.array(m)
        for i in range(q.shape[1]):
            q[:, i] *= m
        return q
    def load_gazesim_data(datapath,subject,run,track,lap,buffer):
        """Load drone data of the gaze sim dataset."""
        fr=Path(datapath)/('s%03d'%subject)/('{}_{}'.format('%02d'%run,track))
        # Load laptimes table.
        l=pd.read_csv(fr/'laptimes.csv')
        # Find lap start and end
        ind=l['lap'].values==lap
        tstart=l.loc[ind,'ts_start'].iloc[0]-buffer
        tend=l.loc[ind,'ts_end'].iloc[0]
        # Load drone data
        d=pd.read_csv(fr/'drone.csv')
        # Rename columns
        ndict={
            'ts':'t',
            'PositionX':'px',
            'PositionY':'py',
            'PositionZ':'pz',
            'rot_x_quat':'qx',
            'rot_y_quat':'qy',
            'rot_z_quat':'qz',
            'rot_w_quat':'qw',
            'VelocityX':'vx',
            'VelocityY':'vy',
            'VelocityZ':'vz',
            'AccX':'ax',
            'AccY':'ay',
            'AccZ':'az',
            'AngularX':'wx',
            'AngularY':'wy',
            'AngularZ':'wz',
            'TrackProgress':'progress',
            'throttle_thrust [N]':'throttle',
            'roll_rate [rad/s]':'roll',
            'pitch_rate [rad/s]':'pitch',
            'yaw_rate [rad/s]':'yaw',
        }
        # Compute sampling rate
        d=d.rename(columns=ndict)
        # Fix the rotation quaternion.
        d.at[:,('qx','qy','qz','qw')]=fix_quaternion(d.loc[:,('qx','qy','qz','qw')].values)
        # Segment to current lap.
        ind=(d['t'].values>=tstart) & (d['t'].values<=tend)
        d=d.loc[ind,:]
        d['t']=d['t'].values-d['t'].iloc[0]
        return d
    d=load_gazesim_data(datapath,subject,run,track,lap,buffer)

    # Print sampling rates.
    print('visatt, reference, SR={:.3f} Hz'.format(1 / np.median(np.diff(r['t'].values))))
    print('visatt, trajectory, SR={:.3f} Hz'.format(1 / np.median(np.diff(t['t'].values))))
    print('gazsim, drone, SR={:.3f} Hz'.format(1 / np.median(np.diff(d['t'].values))))

    # Compare visatt-reference with gazesim-drone trajectory.
    def plot_trajectory_comparison(r,d):
        """Plot a drone state comparison between visatt reference.csv (r) and gazesim drone.csv (d)"""
        plt.figure()
        plt.gcf().set_figwidth(8)
        plt.gcf().set_figheight(16)

        nx=6
        ny=1
        ni=0

        ni+=1
        plt.subplot(nx,ny,ni)
        plt.plot(r.px,r.py,label='visatt')
        plt.plot(d.px,d.py,label='gazesim')
        plt.xlabel('px')
        plt.ylabel('py')
        plt.legend()

        ni+=1
        plt.subplot(nx,ny,ni)
        plt.plot(r.t,r.px,'r-',label='visatt px')
        plt.plot(r.t,r.py,'g-',label='visatt py')
        plt.plot(r.t,r.pz,'b-',label='visatt pz')
        plt.plot(d.t,d.px,'r--',label='gazesim px')
        plt.plot(d.t,d.py,'g--',label='gazesim py')
        plt.plot(d.t,d.pz,'b--',label='gazesim pz')
        plt.xlabel('t')
        plt.ylabel('position [m]')
        plt.legend()

        ni+=1
        plt.subplot(nx,ny,ni)
        plt.plot(r.t,r.qx,'r-',label='visatt qx')
        plt.plot(r.t,r.qy,'g-',label='visatt qy')
        plt.plot(r.t,r.qz,'b-',label='visatt qz')
        plt.plot(r.t,r.qw,'k-',label='visatt qw')
        plt.plot(d.t,d.qx,'r--',label='gazesim qx')
        plt.plot(d.t,d.qy,'g--',label='gazesim qy')
        plt.plot(d.t,d.qz,'b--',label='gazesim qz')
        plt.plot(d.t,d.qw,'k--',label='gazesim qw')
        plt.xlabel('t')
        plt.ylabel('rotation [quaternion]')
        plt.legend()

        ni+=1
        plt.subplot(nx,ny,ni)
        plt.plot(r.t,r.vx,'r-',label='visatt vx')
        plt.plot(r.t,r.vy,'g-',label='visatt vy')
        plt.plot(r.t,r.vz,'b-',label='visatt vz')
        plt.plot(d.t,d.vx,'r--',label='gazesim vx')
        plt.plot(d.t,d.vy,'g--',label='gazesim vy')
        plt.plot(d.t,d.vz,'b--',label='gazesim vz')
        plt.xlabel('t')
        plt.ylabel('velocity [m/s]')
        plt.legend()

        ni+=1
        plt.subplot(nx,ny,ni)
        plt.plot(r.t,r.ax,'r-',label='visatt ax')
        plt.plot(r.t,r.ay,'g-',label='visatt ay')
        plt.plot(r.t,r.az,'b-',label='visatt az')
        plt.plot(d.t,d.ax,'r--',label='gazesim ax')
        plt.plot(d.t,d.ay,'g--',label='gazesim ay')
        plt.plot(d.t,d.az,'b--',label='gazesim az')
        plt.xlabel('t')
        plt.ylabel('acceleration [m/s/s]')
        plt.legend()

        ni+=1
        plt.subplot(nx,ny,ni)
        plt.plot(r.t,r.wx,'r-',label='visatt wx')
        plt.plot(r.t,r.wy,'g-',label='visatt wy')
        plt.plot(r.t,r.wz,'b-',label='visatt wz')
        plt.plot(d.t,d.wx,'r--',label='gazesim wx')
        plt.plot(d.t,d.wy,'g--',label='gazesim wy')
        plt.plot(d.t,d.wz,'b--',label='gazesim wz')
        plt.xlabel('t')
        plt.ylabel('angular [rad/s]')
        plt.legend()

        plt.tight_layout()
    plot_trajectory_comparison(r,d)
    # plt.show()

    def plot_command_comparison(d,t):
        """Plot a comparison of control commands between gazesim-drone and visatt-trajectory."""
        plt.figure()
        plt.gcf().set_figwidth(10)
        plt.gcf().set_figheight(5)

        nx = 1
        ny = 1
        ni = 0

        ni += 1
        plt.subplot(nx, ny, ni)
        plt.plot(d.t, d.throttle, 'k--', label='gazesim throttle')
        plt.plot(d.t, d.roll, 'r--', label='gazesim roll')
        plt.plot(d.t, d.pitch, 'g--', label='gazesim pitch')
        plt.plot(d.t, d.yaw, 'b--', label='gazesim yaw')
        plt.plot(t.t, t.throttle_nw, 'k-', label='visatt nw throttle')
        plt.plot(t.t, t.roll_nw, 'r-', label='visatt nw roll')
        plt.plot(t.t, t.pitch_nw, 'g-', label='visatt nw pitch')
        plt.plot(t.t, t.yaw_nw, 'b-', label='visatt nw yaw')
        plt.xlabel('t')
        plt.ylabel('control command')
        plt.legend()
    plot_command_comparison(d,t)
    plt.show()