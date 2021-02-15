import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3D
from scipy.spatial.transform import Rotation

def trajectory_from_flightmare(filepath: str) -> pd.DataFrame():
    """Returns a trajectory dataframe with standard headers from a flightmare
    log filepath."""
    ndict = {
        'time-since-start [s]': 't',
        'position_x [m]': 'px',
        'position_y [m]': 'py',
        'position_z [m]': 'pz',
        'rotation_x [quaternion]': 'qx',
        'rotation_y [quaternion]': 'qy',
        'rotation_z [quaternion]': 'qz',
        'rotation_w [quaternion]': 'qw',
        'velocity_x': 'vx',
        'velocity_y': 'vy',
        'velocity_z': 'vz',
        'acceleration_x': 'ax',
        'acceleration_y': 'ay',
        'acceleration_z': 'az',
        'omega_x': 'wx',
        'omega_y': 'wy',
        'omega_z': 'wz'
        }
    if os.path.exists(filepath) is False:
        return pd.DataFrame([])
    else:
        df = pd.read_csv(filepath)
        # Rename columns according to the name dictionairy.
        for search_name, replacement_name in ndict.items():
            for column_name in df.columns:
                if column_name.find(search_name) != -1:
                    df = df.rename(columns={column_name: replacement_name})
        # Sort columns using found dictionairy entries first and unknown last.
        known_names = [name for name in ndict.values() if name in df.columns]
        unknown_names = [name for name in df.columns if name not in known_names]
        sorted_names = known_names.copy()
        sorted_names.extend(unknown_names)
        df = df.loc[:, sorted_names]
        # Sort values by timestamp
        df = df.loc[:, sorted_names]
        df = df.sort_values(by=['t'])
        return df


def plot_trajectory(
        px: np.ndarray([]), py: np.ndarray([]), pz: np.ndarray([])=np.array([]),
        qx: np.ndarray([])=np.array([]), qy: np.ndarray([])=np.array([]),
        qz: np.ndarray([])=np.array([]), qw: np.ndarray([])=np.array([]),
        c: str=None, ax: plt.axis()=None, axis_length: float=1.,
        ) -> plt.axis():
    """Returns an axis handle for a 2D or 3D trajectory based on position and
    rotation data."""
    # Check if the plot is 2D or 3D.
    if pz.shape[0] == 0:
        is_two_d = True
    else:
        is_two_d = False
    # Make a new figure if no axis was provided.
    if ax is None:
        fig = plt.figure()
        if is_two_d:
            ax = fig.add_subplot(1, 1, 1)
        else:
            ax = fig.add_subplot(1, 1, 1, projection='3d')
    # Plot the flightpath
    if is_two_d:
        if c is None:
            ax.plot(px, py)
        else:
            ax.plot(px, py, color=c)
    else:
        if c is None:
            ax.plot(px, py, pz)
        else:
            ax.plot(px, py, pz, color=c)
    # Plot 3D quadrotor rotation
    if not is_two_d:
        if ((qx.shape[0] > 0) and (qy.shape[0] > 0) and (qz.shape[0] > 0)
                and (qw.shape[0] > 0)):
            for primitive, color in [((1, 0, 0), 'r'),
                                     ((0, 1, 0), 'g'),
                                     ((0, 0, 1), 'b')]:
                p0 = np.hstack((px.reshape(-1, 1), np.hstack((py.reshape(-1, 1),
                                                              pz.reshape(-1, 1)))))
                q = np.hstack((qx.reshape(-1, 1),
                               np.hstack((qy.reshape(-1, 1),
                                          np.hstack((qz.reshape(-1, 1),
                                                     qw.reshape(-1, 1)))))))
                p1 = p0 + Rotation.from_quat(q).apply(np.array(primitive)
                                                      * axis_length)
                for i in (range(p0.shape[0])):
                    ax.plot([p0[i, 0], p1[i, 0]], [p0[i, 1], p1[i, 1]],
                            [p0[i, 2], p1[i, 2]], color=color)
    return ax


def format_trajectory_figure(
        ax: plt.axis(), xlims: tuple=(), ylims: tuple=(), zlims: tuple=(),
        xlabel: str='', ylabel: str='', zlabel: str='', title: str='',
        ) -> plt.axis():
    """Apply limits, labels, and title formatting for a supplied figure axis."""
    if len(xlims) > 0:
        ax.set_xlim(xlims)
    if len(ylims) > 0:
        ax.set_ylim(ylims)
    if len(zlims) > 0:
        ax.set_zlim(zlims)
    if len(xlabel) > 0:
        ax.set_xlabel(xlabel)
    if len(ylabel) > 0:
        ax.set_ylabel(ylabel)
    if len(zlabel) > 0:
        ax.set_zlabel(zlabel)
    if len(title) > 0:
        ax.set_title(title)
    return ax

