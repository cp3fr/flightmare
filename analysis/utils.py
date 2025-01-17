import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import yaml

from mpl_toolkits.mplot3d.art3d import Line3D
from multiprocessing import Pool
from pathlib import Path
from pprint import pprint
from scipy.spatial.transform import Rotation
from scipy.stats import iqr
from shapely.geometry import LineString
from shutil import copyfile
from skspatial.objects import Vector, Points, Line, Point, Plane
from skspatial.plotting import plot_3d


class Checkpoint:
    """
    Checkpoint object represented as 2D surface in 3D space with x-axis
    pointing in the direction of flight/desired passing direction.

    Contains useful methods for determining distance of points,
    and intersections with the plane.
    """
    def __init__(
            self,
            df,
            dims=None,
            dtype='liftoff',
            ):
        #set variable names according to the type of data
        if dtype=='liftoff':
            position_varnames = ['px', 'py', 'pz']
            rotation_varnames = ['qx', 'qy', 'qz', 'qw']
            dimension_varnames = ['dy', 'dz']
            dimension_scaling_factor = 1.
        if dtype=='gazesim':
            position_varnames = ['pos_x', 'pos_y', 'pos_z']
            rotation_varnames = ['rot_x_quat', 'rot_y_quat', 'rot_z_quat', 'rot_w_quat']
            dimension_varnames = ['dim_y', 'dim_z']
            dimension_scaling_factor = 2.5
        else: #default
            position_varnames = ['px', 'py', 'pz']
            rotation_varnames = ['qx', 'qy', 'qz', 'qw']
            dimension_varnames = ['dy', 'dz']
            dimension_scaling_factor = 1.
        #current gate center position
        p = df[position_varnames].values.flatten()
        #current gate rotation quaterion
        q = df[rotation_varnames].values.flatten()
        #if no width and height dimensions were specified
        if dims is None:
            dims = (df[dimension_varnames[0]] * dimension_scaling_factor,
                    df[dimension_varnames[1]] * dimension_scaling_factor)
        #half width and height of the gate
        hw = dims[0] / 2. #half width
        hh = dims[1] / 2. #half height
        #assuming gates are oriented in the direction of flight: x=forward, y=left, z=up
        proto = np.array([[ 0.,  0.,  0.,  0.,  0.],  # assume surface, thus no thickness along x axis
                          [ hw, -hw, -hw,  hw,  hw],
                          [ hh,  hh, -hh, -hh,  hh]])
        self._corners = (Rotation.from_quat(q).apply(proto.T).T + p.reshape(3,
        1)).astype(float)
        self._center = p
        self._rotation = q
        self._normal = Rotation.from_quat(q).apply(np.array([1, 0, 0]))
        self._width = dims[0]
        self._height = dims[1]
        #1D minmax values
        self._x = np.array([np.min(self._corners[0, :]), np.max(self._corners[0, :])])
        self._y = np.array([np.min(self._corners[1, :]), np.max(self._corners[1, :])])
        self._z = np.array([np.min(self._corners[2, :]), np.max(self._corners[2, :])])
        #2D line representations of gate horizontal axis
        self._xy = LineString([ ((np.min(self._corners[0, :])), np.min(self._corners[1, :])),
                                ((np.max(self._corners[0, :])), np.max(self._corners[1, :]))])
        self._xz = LineString([((np.min(self._corners[0, :])), np.min(self._corners[2, :])),
                               ((np.max(self._corners[0, :])), np.max(self._corners[2, :]))])
        self._yz = LineString([((np.min(self._corners[1, :])), np.min(self._corners[2, :])),
                               ((np.max(self._corners[1, :])), np.max(self._corners[2, :]))])
        #plane representation
        center_point = Point(list(self._center))
        normal_point = Point(list(self._center + self._normal))
        normal_vector = Vector.from_points(center_point, normal_point)
        self.plane = Plane(point=center_point, normal=normal_vector)

        self.x_axis = Line(point=self._corners[:, 0],
                      direction=self._corners[:, 1] - self._corners[:, 0])
        self.y_axis = Line(point=self._corners[:, 0],
                      direction=self._corners[:, 3] - self._corners[:, 0])
        self.length_x_axis = self.x_axis.direction.norm()
        self.length_y_axis = self.y_axis.direction.norm()

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def corners(self):
        return self._corners

    @property
    def center(self):
        return self._center

    @property
    def rotation(self):
        return self._rotation

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def xy(self):
        return self._xy

    @property
    def xz(self):
        return self._xz

    @property
    def yz(self):
        return self._yz

    def detect_pass(
            self,
            t: '(n,) np.array: timestamps in seconds',
            p: '(n, 3) np.array: drone positions in world frame',
            debug: 'Whether to show debugging plot'=False,
            ) -> '(n,) list: checkpoint crossing timestamps':
        """
        Returns a list of checkpoint crossing timestamps from given drone
        position (p) and time (t).

        It uses coordinate transformation of drone data into the checkpoint
        frame.
        """

        # Drone positions in world frame.
        p_W = np.hstack((p, np.ones((p.shape[0], 1)))).astype(np.float64)

        # Transform checkpoint in world frame.
        T_W_C = np.vstack((
            np.hstack((
                Rotation.from_quat(self._rotation).as_matrix(),
                self._center.reshape((-1, 1)),
            )),
            np.array(([[0., 0., 0., 1.]]))
        )).astype(np.float64)

        # Transform world in checkpoint frame.
        T_C_W = np.linalg.inv(T_W_C)

        # Drone positions in checkpoint frame.
        p_C = (T_C_W @ p_W.T).T[:, :3]

        # Plot and top view overlay of drone and checkpoint in different
        # coordinate frames.
        if debug:
            plt.figure()
            plt.plot(p_W[:, 0],
                     p_W[:, 1],
                     'b',
                     label='p_W (drone position in world frame)')
            plt.plot(self._center[0],
                     self._center[1],
                     'bo',
                     label='c_W (checkpoint center in world frame)')
            plt.plot(p_C[:, 0],
                     p_C[:, 1],
                     'g',
                     label='p_C (drone position in checkpoint frame)')
            plt.plot(0,
                     0,
                     'go',
                     label='c_C (checkpoint position in checkpoint frame)')
            plt.xlabel('Position X [m]')
            plt.ylabel('Position Y [m]')
            plt.legend()
            plt.show()

        # Signed distance from gate, is identical to the x-axis values of
        # drone positions in checkpoint frame.
        # Remember: desired direction of flight is from negative x to positive x
        distance = p_C[:, 0]

        # Detect checkpoint "plane" crossings, i.e. when signed distance
        # changes from negative to positive.
        # Note. This can happen at arbitrary distances from the checkpoint
        # center.
        ind0 = distance > 0
        ind1 = np.hstack((True, ind0[:-1]))
        ind_crossing = (ind0 == True) & (ind1 == False)

        # Check whether "plane" crossings fall within the boundaries of
        # the checkpoint.
        # We assume checkpoint ins a plane, thus x-axis dimension is zero
        # Width is the checkpoint dimension across y-axis
        # Height is the checkpoint dimension across z-axis
        ind_within_borders = (
                (np.abs(p_C[:, 1]) <= (self._width / 2.)) &
                (np.abs(p_C[:, 2]) <= (self._height / 2.)))

        # Finally, find "checkpoint crossings", by selecting those events
        # that have a "plane crossing" somewhere and where positions fall
        # within the checkpoint borders.
        ind_events = ind_crossing & ind_within_borders

        # Return checkpoint pass events as a list of timestamps.
        return list(t[ind_events])

    def get_distance(
            self,
            p,
            ):
        return self.plane.distance_point(p)

    def get_signed_distance(
            self,
            p,
            ):
        return self.plane.distance_point_signed(p)

    def check_within_borders(
            self,
            p_W,
            ):
        """
        Check if point is within checkpoint borders.
        """
        # Project point into the checkpoint frame
        p_C = Rotation.from_quat(self._rotation).apply((p_W.astype(
            np.float64) - self._center.astype(np.float64)))
        # Check if withing the y and z dimensions of the checkpoint planed
        if ((np.abs(p_C[1]) <= (self.width / 2)) &
            (np.abs(p_C[2]) <= (self.height / 2))):
            return True
        else:
            return False


    def intersect(
            self,
            p0: np.array,
            p1: np.array,
            ) -> tuple:
        """
        Returns 2D and 3D intersection points with the gate surface and a
        line from two given points (p0, p1).
        """
        # Initialize relevant variables
        point_2d = None
        point_3d = None
        _x = None
        _y = None
        _z = None
        count = 0
        #only proceed if no nan values
        if (np.sum(np.isnan(p0).astype(int))==0) & (np.sum(np.isnan(p1).astype(int))==0):
            #line between start end end points
            line_xy = LineString([(p0[0], p0[1]),
                                  (p1[0], p1[1])])
            line_xz = LineString([(p0[0], p0[2]),
                                  (p1[0], p1[2])])
            line_yz = LineString([(p0[1], p0[2]),
                                  (p1[1], p1[2])])
            if self.xy.intersects(line_xy):
                count += 1
                _x, _y = [val for val in self.xy.intersection(line_xy).coords][0]
            if self.xz.intersects(line_xz):
                count += 1
                _x, _z = [val for val in self.xz.intersection(line_xz).coords][0]
            if self.yz.intersects(line_yz):
                count += 1
                _y, _z = [val for val in self.yz.intersection(line_yz).coords][0]
            #at least two of the three orthogonal lines need to be crossed
            if count > 1:
                point_3d = np.array([_x, _y, _z])
                point_2d = self.point2d(point_3d)
        return point_2d, point_3d

    def point2d(
            self,
            p: np.array,
            ) -> np.array:
        """
        Returns normalized [0-1 range] 2D coordinates of the intersection
        point within gate surface.

        The origin is the upper left corner (1st corner point)
        X-axis is to the right
        Y-axis is down
        """
        # Project the 3D intersection point onto the x and y surface axies.
        px_projected = self.x_axis.project_point(p)
        py_projected = self.y_axis.project_point(p)
        length_px_projected = self.x_axis.point.distance_point(px_projected)
        length_py_projected = self.y_axis.point.distance_point(py_projected)
        # Return the normalized 2D projection of the intersection point.
        return np.array([length_px_projected / self.length_x_axis,
                         length_py_projected / self.length_y_axis])


def get_wall_colliders(
        dims: tuple=(1, 1, 1),
        center: tuple=(0, 0, 0)
        ) -> list:
    """Returns a list of 'Gate' objects representing the walls of a race
    track in 3D.
        dims: x, y, z dimensions in meters
        denter: x, y, z positions of the 3d volume center"""
    objWallCollider = []

    _q = (Rotation.from_euler('y', [np.pi/2]) * Rotation.from_quat(np.array([0, 0, 0, 1]))).as_quat().flatten()

    objWallCollider.append(Checkpoint(pd.DataFrame({'px': [center[0]],
                                                    'py': [center[1]],
                                                    'pz': [center[2] - dims[2]/2],
                                                    'qx': [_q[0]],
                                                    'qy': [_q[1]],
                                                    'qz': [_q[2]],
                                                    'qw': [_q[3]],
                                                    'dx': [0],
                                                    'dy': [dims[1]],
                                                    'dz': [dims[0]]},
                                                    index=[0]).iloc[0],
                                                    dims=(dims[1], dims[0])))

    _q = (Rotation.from_euler('y', [-np.pi/2]) * Rotation.from_quat(np.array([0, 0, 0, 1]))).as_quat().flatten()
    objWallCollider.append(Checkpoint(pd.DataFrame({'px': center[0],
                                                    'py': center[1],
                                                    'pz' : center[2] + dims[2] / 2,
                                                    'qx': _q[0],
                                                    'qy':_q[1],
                                                    'qz':_q[2],
                                                    'qw':_q[3],
                                                    'dx':0,
                                                    'dy':dims[1],
                                                    'dz':dims[0]},
                                                     index=[0]).iloc[0],
                                                     dims=(dims[1], dims[0])))

    _q = np.array([0, 0, 0, 1])
    objWallCollider.append(Checkpoint(pd.DataFrame({'px': center[0] + dims[0]/ 2,
                                                    'py': center[1],
                                                    'pz' : center[2],
                                                    'qx': _q[0],
                                                    'qy':_q[1],
                                                    'qz':_q[2],
                                                    'qw':_q[3],
                                                    'dx':0,
                                                    'dy':dims[1],
                                                    'dz':dims[2]},
                                                    index=[0]).iloc[0],
                                                    dims=(dims[1], dims[2])))

    _q = (Rotation.from_euler('z', [np.pi]) * Rotation.from_quat(np.array([0, 0, 0, 1]))).as_quat().flatten()
    objWallCollider.append(Checkpoint(pd.DataFrame({'px': center[0] - dims[0] / 2,
                                                    'py': center[1],
                                                    'pz' : center[2],
                                                    'qx': _q[0],
                                                    'qy':_q[1],
                                                    'qz':_q[2],
                                                    'qw':_q[3],
                                                    'dx':0,
                                                    'dy':dims[1],
                                                    'dz':dims[2]},
                                                    index=[0]).iloc[0],
                                                    dims=(dims[1], dims[2])))

    _q = (Rotation.from_euler('z', [np.pi/2]) * Rotation.from_quat(np.array([0, 0, 0, 1]))).as_quat().flatten()
    objWallCollider.append(Checkpoint(pd.DataFrame({'px': center[0],
                                                    'py': center[1] + dims[1] / 2,
                                                    'pz' : center[2],
                                                    'qx': _q[0],
                                                    'qy':_q[1],
                                                    'qz':_q[2],
                                                    'qw':_q[3],
                                                    'dx':0,
                                                    'dy':dims[1],
                                                    'dz':dims[2]},
                                                    index=[0]).iloc[0],
                                                    dims=(dims[0], dims[2])))

    _q = (Rotation.from_euler('z', [-np.pi/2]) * Rotation.from_quat(np.array([0, 0, 0, 1]))).as_quat().flatten()
    objWallCollider.append(Checkpoint(pd.DataFrame({'px': center[0],
                                                    'py': center[1] - dims[1] / 2,
                                                    'pz' : center[2],
                                                    'qx': _q[0],
                                                    'qy':_q[1],
                                                    'qz':_q[2],
                                                    'qw':_q[3],
                                                    'dx':0,
                                                    'dy':dims[1],
                                                    'dz':dims[2]},
                                                    index=[0]).iloc[0],
                                                    dims=(dims[0], dims[2])))
    return objWallCollider


def make_path(
        path: str
        ) -> bool():
    """Make (nested) folders, if not already existent, from provided path."""
    if isinstance(path, str):
        p=Path(path)
    else:
        p=path
    if not p.exists():
        p.mkdir(parents=True,exist_ok=True)


def trajectory_from_logfile(
        filepath: 'str/Path'
        ) -> pd.DataFrame():
    """Returns a trajectory dataframe with standard headers from a flightmare
    log filepath."""
    if isinstance(filepath,str):
        filepath=Path(filepath)
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
    if filepath.exists():
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
    else:
        return pd.DataFrame([])


def plot_trajectory(
        px: np.ndarray([]),
        py: np.ndarray([]),
        pz: np.ndarray([])=np.array([]),
        qx: np.ndarray([])=np.array([]),
        qy: np.ndarray([])=np.array([]),
        qz: np.ndarray([])=np.array([]),
        qw: np.ndarray([])=np.array([]),
        c: str=None,
        ax: plt.axis()=None,
        axis_length: float=1.,
        axis_colors: list=['r', 'g', 'b'],
        ) -> plt.axis():
    """Returns an axis handle for a 2D or 3D trajectory based on position and
    rotation data."""
    # Check if the plot is 2D or 3D.
    if isinstance(axis_colors, str):
        axis_colors = list(axis_colors)
    while len(axis_colors) < 3:
        axis_colors.append(axis_colors[-1])
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
            for primitive, color in [((1, 0, 0), axis_colors[0]),
                                     ((0, 1, 0), axis_colors[1]),
                                     ((0, 0, 1), axis_colors[2])]:
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
        ax: plt.axis(),
        xlims: tuple=(),
        ylims: tuple=(),
        zlims: tuple=(),
        xlabel: str='',
        ylabel: str='',
        zlabel: str='',
        title: str='',
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


def get_pass_collision_events(
        filepath_trajectory: str,
        filepath_track: str,
        gate_inner_dimensions: tuple=(2.5, 2.5),
        gate_outer_dimensions: tuple=(3.5, 3.5),
        wall_collider_dimensions: tuple=(66, 36, 9),
        wall_collider_center: tuple=(0, 0, 4.85)
        ) -> pd.DataFrame():
    """Returns an events dataframe of pass and collision events from given
    trajectory and track filepaths."""
    if isinstance(filepath_trajectory,str):
        filepath_trajectory=Path(filepath_trajectory)
    if isinstance(filepath_track,str):
        filepath_track=Path(filepath_track)
    # Load race track information.
    T = pd.read_csv(filepath_track)
    # Define checkpoints and colliders.
    gate_checkpoints = [
        Checkpoint(T.iloc[i], dims=gate_inner_dimensions) for i in
        range(T.shape[0])]
    gate_colliders = [
        Checkpoint(T.iloc[i], dims=gate_outer_dimensions) for i in
        range(T.shape[0])]
    wall_colliders = get_wall_colliders(dims=wall_collider_dimensions,
                                        center=wall_collider_center)
    # Load a trajectory
    D = pd.read_csv(filepath_trajectory)
    t = D['t'].values
    p = D.loc[:, ('px', 'py', 'pz')].values
    px = D['px'].values
    py = D['py'].values
    pz= D['pz'].values
    # Detect checkpoint passing and collision events.
    events = {}
    """
    for key, objects in [('gate_pass', gate_checkpoints),
                         ('gate_collision', gate_colliders),
                         ('wall_collision', wall_colliders)
                        ]:
    """
    for key, objects in [('gate_collision', gate_colliders),
                         ('gate_pass', gate_checkpoints),
                         ('wall_collision', wall_colliders)
                         ]:
        for id in range(len(objects)):
            object = objects[id]
            # for timestamp in detect_checkpoint_pass(t, px, py, pz, object):
            for timestamp in object.detect_pass(t=t,p=p):
                if not ((key == 'gate_collision') and (
                        timestamp in events.keys())):
                    test_x = px[t == timestamp][0]
                    test_y = py[t == timestamp][0]
                    test_z = pz[t == timestamp][0]
                    events[timestamp] = (key, id, test_x, test_y, test_z)
    # Make events dataframe
    E = pd.DataFrame({
        't': [
            k
            for k in sorted(events)],
        'object-id': [
            events[k][1]
            for k in sorted(events)],
        'object-name': [
            events[k][0].split('_')[0]
            for k in sorted(events)],
        'is-collision': [
            int(events[k][0].find('collision') > 0)
            for k in sorted(events)],
        'is-pass':  [
            int(events[k][0].find('pass') > 0)
            for k in sorted(events)],
        'px': [
            events[k][2]
            for k in sorted(events)],
        'py': [
            events[k][3]
            for k in sorted(events)],
        'pz': [
            events[k][4]
            for k in sorted(events)],
        },
        index=list(range(len(events)))
        )
    return E


def resample_dataframe(
        df: pd.DataFrame,
        sr: float,
        varname: str='t',
        ) -> pd.DataFrame:
    """
    Resample a dataframe.
    """
    x = df[varname].values
    xn = np.arange(x[0], x[-1], 1/sr)
    rdict = {}
    for n in df.columns:
        rdict[n] = np.interp(xn, x, df[n].values)
    return pd.DataFrame(rdict, index=list(range(xn.shape[0])))


def extract_performance_features(
        filepath_trajectory: str,
        filepath_events: str,
        filepath_reference: str=None,
        colliders: list=['gate', 'wall'],
        debug: bool=False,
        ) -> pd.DataFrame():
    """
    Compute performance features for a given network trajectory, considering
    comparison to reference trajectory, checkpoint passing, and collision
    events.
    """
    # Load a trajectory
    D = pd.read_csv(filepath_trajectory)
    sr = 1/np.nanmedian(np.diff(D['t'].values))
    D = resample_dataframe(df=D, sr=sr, varname='t')
    # Load the reference trajectory
    R = pd.read_csv(filepath_reference)
    R = resample_dataframe(df=R, sr=sr, varname='t')
    # Load events
    E = pd.read_csv(filepath_events)
    # Remove some collision events, depending on which colliders shall be used.
    ind = E['is-collision'].values == 0
    for n in  colliders:
        ind[(E['is-collision'].values == 1) &
            (E['object-name'].values == n)] = True
    E = E.loc[ind, :]
    # Remove initial buffer time.
    buffer = float(filepath_trajectory.as_posix().split('buffer')[-1].split(
        '/')[0]) / 10
    ind=R['t'].values>buffer
    R=R.loc[ind,:]
    ind=D['t'].values>buffer
    D=D.loc[ind, :]
    ind=E['t'].values>buffer
    E=E.loc[ind,:]
    # Check if network (online) or mpc (offline) control mode,
    x = D['throttle_mpc'].values
    nmpc=np.sum(np.isnan(x)==False)
    ntotal=x.shape[0]
    if nmpc/ntotal>0.8:
        c='mpc'
    else:
        c = 'nw'
    # Compute control command prediction performance for mpc control mode.
    mpc_nw_dict = {}
    if c=='mpc':
        network_used = 0
        for n in ['throttle', 'roll', 'pitch', 'yaw']:
            mpc_nw_dict.setdefault(n, {})
            x=(D['{}_mpc'.format(n)].values-D['{}_nw'.format(n)].values)
            mpc_nw_dict[n]['l1']=np.nanmean(np.abs(x))
            mpc_nw_dict[n]['mse']=np.nanmean(np.power(x, 2))
            mpc_nw_dict[n]['l1-median']=np.nanmedian(np.abs(x))
            mpc_nw_dict[n]['mse-median']=np.nanmedian(np.power(x, 2))
    else:
        network_used = 1
        for n in ['throttle', 'roll', 'pitch', 'yaw']:
            mpc_nw_dict.setdefault(n, {})
            for m in ['l1', 'mse', 'l1-median', 'mse-median']:
                mpc_nw_dict[n][m]=np.nan
    # Determine start and end time
    t_start=D['t'].iloc[0]
    t_end=D['t'].iloc[-1]
    ind=E['is-collision'].values==1
    if np.sum(ind)>0:
        t_end=E.loc[ind, 't'].values[0]
    # Get performance metrics within the start and end time window
    # ..total number of passed gates.
    ind = ((E['t'].values >= t_start) &
           (E['t'].values <= t_end))
    num_gates_passed = np.sum(E.loc[ind, 'is-pass'].values)
    # ..number of collisions.
    ind = ((E['t'].values >= t_start) &
           (E['t'].values <= t_end) &
           (E['is-collision'].values == 1))
    num_collisions = np.sum(ind)
    # ..number of individual gate passes.
    num_passes = {}
    for i in range(10):
        ind = ((E['t'].values >= t_start) &
               (E['t'].values <= t_end) &
               (E['object-id'].values == i))
        num_passes[i] = np.sum(E.loc[ind, 'is-pass'].values)
    # Distance and Duration features (considering all time the network was in
    # control)
    if t_end > t_start:
        ind = ((D['t'].values>=t_start) & (D['t'].values<=t_end))
        total_distance = np.sum(np.linalg.norm(np.diff(D.loc[ind, ('px', 'py',
            'pz')].values,axis=0),axis=1))
        total_duration = t_end - t_start
    else:
        total_distance = 0.
        total_duration = 0.
    # Compute deviation from reference, but only after having passed the
    # center gate
    # default values
    deviation_from_reference = 0.
    median_deviation = 0.
    iqr_deviation = 0.
    t_min = np.max([t_start, D['t'].iloc[0], R['t'].iloc[0]])
    t_max = np.min([t_end, D['t'].iloc[-1], R['t'].iloc[-1]])
    if t_max>t_min:
        ind = (D['t'].values>=t_min) & (D['t'].values<=t_max)
        D = D.iloc[ind, :]
        ind = (R['t'].values>=t_min) & (R['t'].values<=t_max)
        R = R.iloc[ind, :]
        if D.shape[0] > R.shape[0]:
            D = D.iloc[:R.shape[0], :]
        elif R.shape[0] > D.shape[0]:
            R = R.iloc[:D.shape[0], :]
        if (D.shape[0]>0) & (R.shape[0]>0):
            p = D.loc[:, ('px', 'py', 'pz')].values
            pref = R.loc[:, ('px', 'py', 'pz')].values
            deviation_from_reference = np.linalg.norm(p-pref, axis=1)
            median_deviation = np.nanmedian(deviation_from_reference)
            iqr_deviation = iqr(deviation_from_reference)
    # Save the features to pandas dataframe
    outdict = {
        't_start': t_start,
        't_end': t_end,
        'flight_time': total_duration,
        'travel_distance': total_distance,
        'median_path_deviation' : median_deviation,
        'iqr_path_deviation' : iqr_deviation,
        'num_gates_passed': num_gates_passed,
        'num_pass_gate0': num_passes[0],
        'num_pass_gate1': num_passes[1],
        'num_pass_gate2': num_passes[2],
        'num_pass_gate3': num_passes[3],
        'num_pass_gate4': num_passes[4],
        'num_pass_gate5': num_passes[5],
        'num_pass_gate6': num_passes[6],
        'num_pass_gate7': num_passes[7],
        'num_pass_gate8': num_passes[8],
        'num_pass_gate9': num_passes[9],
        'num_collisions': num_collisions,
        'network_used': network_used,
        }
    for k1 in sorted(mpc_nw_dict):
        for k2 in sorted(mpc_nw_dict[k1]):
            outdict['{}_error_{}'.format(k1, k2)] = mpc_nw_dict[k1][k2]
    outdict['filepath'] = filepath_trajectory
    P = pd.DataFrame(outdict, index=[0])
    # Debug:check the data
    if debug:
        print('-------------------------')
        print('DEBUG')
        print('-------------------------')
        print(filepath_trajectory)
        print(D.columns)
        print(E.columns)
        print(E)
        print('mpc samples: {}/{}'.format(nmpc, ntotal))
        print(P.loc[:, ('throttle_error_l1',
                        'throttle_error_l1-median', 'throttle_error_mse',
                        'throttle_error_mse-median')])
        plt.figure()
        plt.gcf().set_figwidth(15)
        plt.subplot(2, 1, 1)
        plt.plot(D.px, D.py, label='nw')
        plt.plot(R.px, R.py, label='reference')
        plt.title('DEBUG:\n{}'.format(filepath_trajectory))
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(D['t'], D['throttle_mpc'], label='throttle_mpc')
        plt.plot(D['t'], D['throttle_nw'], label='throttle_nw')
        plt.xlabel('t')
        plt.ylabel('throttle')
        plt.legend()
        plt.show()

    return P


def plot_state(
        filepath_trajectory: str,
        filepath_features: str,
        filepath_reference: str=None,
        ax: plt.axis=None,
        ) -> pd.DataFrame():
    """
    Plot drone state variables
    """
    features = pd.read_csv(filepath_features)
    t_start = features.t_start.iloc[0]
    t_end = features.t_end.iloc[0]
    trajectory = pd.read_csv(filepath_trajectory)
    reference = pd.read_csv(filepath_reference)
    ind = (trajectory.t.values >= t_start) & (trajectory.t.values <= t_end)
    trajectory = trajectory.iloc[ind, :]
    ind = (reference.t.values >= t_start) & (reference.t.values <= t_end)
    reference = reference.iloc[ind, :]

    if ax == None:
        fig , axs = plt.subplots(4, 1)
        fig.set_figwidth(15)
        fig.set_figheight(20)
    else:
        axs = [ax]

    qt = trajectory.loc[:, ('qx', 'qy', 'qz', 'qw')].values
    qt = Rotation.from_quat(qt).as_quat()

    qr = reference.loc[:, ('qx', 'qy', 'qz', 'qw')].values
    qr = Rotation.from_quat(qr).as_quat()

    iax = 0
    axs[iax].plot(trajectory.t, trajectory.px, 'r-')
    axs[iax].plot(trajectory.t, trajectory.py, 'g-')
    axs[iax].plot(trajectory.t, trajectory.pz, 'b-')
    axs[iax].plot(reference.t, reference.px, 'r--')
    axs[iax].plot(reference.t, reference.py, 'g--')
    axs[iax].plot(reference.t, reference.pz, 'b--')
    axs[iax].set_xlabel('Time [sec]')
    axs[iax].set_ylabel('Position [m]')
    axs[iax].legend(['trajectory px', 'trajectory py', 'trajectory pz',
                     'reference px', 'reference py', 'reference pz',],
                    loc='upper right')
    iax += 1
    axs[iax].plot(trajectory.t, qt[:, 0], 'r-')
    axs[iax].plot(trajectory.t, qt[:, 1], 'g-')
    axs[iax].plot(trajectory.t, qt[:, 2], 'b-')
    axs[iax].plot(trajectory.t, qt[:, 3], 'm-')
    axs[iax].plot(reference.t, qr[:, 0], 'r--')
    axs[iax].plot(reference.t, qr[:, 1], 'g--')
    axs[iax].plot(reference.t, qr[:, 2], 'b--')
    axs[iax].plot(reference.t, qr[:, 3], 'm--')
    axs[iax].set_xlabel('Time [sec]')
    axs[iax].set_ylabel('Rotation [quaternion]')
    axs[iax].legend(['trajectory qx', 'trajectory qy', 'trajectory qz',
                     'trajectory qw',
                     'reference qx', 'reference qy', 'reference qz',
                     'reference qw', ],
                    loc='upper right')

    iax += 1
    axs[iax].plot(trajectory.t, trajectory.vx, 'r-')
    axs[iax].plot(trajectory.t, trajectory.vy, 'g-')
    axs[iax].plot(trajectory.t, trajectory.vz, 'b-')
    axs[iax].plot(reference.t, reference.vx, 'r--')
    axs[iax].plot(reference.t, reference.vy, 'g--')
    axs[iax].plot(reference.t, reference.vz, 'b--')
    axs[iax].set_xlabel('Time [sec]')
    axs[iax].set_ylabel('Velocity [m/s]')
    axs[iax].legend(['trajectory vx', 'trajectory vy', 'trajectory vz',
                     'reference vx', 'reference vy', 'reference vz', ],
                    loc='upper right')

    iax += 1
    axs[iax].plot(trajectory.t, trajectory.throttle_nw, 'k-',label='nw throttle')
    axs[iax].plot(trajectory.t, trajectory.roll_nw, 'r-',label='nw roll')
    axs[iax].plot(trajectory.t, trajectory.pitch_nw, 'g-',label='nw pitch')
    axs[iax].plot(trajectory.t, trajectory.yaw_nw, 'b-',label='nw yaw')
    axs[iax].plot(trajectory.t, trajectory.throttle_mpc, 'k--', label='mpc throttle')
    axs[iax].plot(trajectory.t, trajectory.roll_mpc, 'r--', label='mpc roll')
    axs[iax].plot(trajectory.t, trajectory.pitch_mpc, 'g--', label='mpc pitch')
    axs[iax].plot(trajectory.t, trajectory.yaw_mpc, 'b--', label='mpc yaw')
    ind=trajectory.network_used.values==1
    axs[iax].plot(trajectory.loc[ind,'t'].values, trajectory.loc[ind,'network_used'].values,
        'mo', lw=4, label='network used')
    axs[iax].set_xlabel('Time [sec]')
    axs[iax].set_ylabel('Control commands')
    axs[iax].legend(loc='upper right')


    return axs


def compare_trajectories_3d(
        reference_filepath: str,
        data_path: str=None,
        ax: plt.axis=None,
        ) -> None:
    """
    Comparison of 3D flight trajectories from two given trajectory logfile
    paths, showing 3D poses.
    """
    if ax == None:
        fig, ax = plt.subplots(1, 1)
    # Plot reference, MPC, and network trajectories in 3D
    if data_path:
        ref = trajectory_from_logfile(
            filepath=reference_filepath)
        ref = ref.iloc[np.arange(0, ref.shape[0], 50), :]
        ax = plot_trajectory(
            ref.px.values,
            ref.py.values,
            ref.pz.values,
            c='k',
            ax=ax)
        for w in os.walk(data_path):
            for f in w[2]:
                if (f.find('trajectory.csv') != -1):
                    if f.find('mpc_eval_nw') != -1:
                        color = 'r'
                    else:
                        color = 'b'
                    filepath = os.path.join(w[0], f)
                    print(filepath)
                    df = trajectory_from_logfile(filepath=filepath)
                    print(df.columns)
                    plot_trajectory(df.px, df.py, df.pz, c=color, ax=ax)
        ax = format_trajectory_figure(ax, xlims=(-30, 30), ylims=(-30, 30),
            zlims=(-30, 30), xlabel='px [m]', ylabel='py [m]', zlabel='pz [m]',
            title=data_path)
    return ax


def plot_gates_3d(
        track: pd.DataFrame,
        ax: plt.axis=None,
        color: str='b',
        width: float=4,
        ) -> plt.axis:
    """
    Plot gates as rectangles in 3D.
    """
    if ax is None:
        fig = plt.figure()
        fig.set_figwidth(20)
        fig.set_figheight(20)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
    # loop over gates
    for igate in range(track.shape[0]):
        # gate center
        position = track.loc[:, ('px', 'py', 'pz')].iloc[igate].values
        # gate rotation
        rotation = track.loc[:, ('qx', 'qy', 'qz', 'qw')].iloc[igate].values
        # checkpoint center
        checkpoint_center = position
        # checkpoint size
        checkpoint_size = track.loc[:, ('dx', 'dy', 'dz')].iloc[igate].values
        # loop over axes
        corners = np.empty((0, 3))

        for y, z in [
            (-1, 1),
            (1, 1),
            (1, -1),
            (-1, -1),
            (-1, 1),
            ]:
                # determine gate corner by: 1. add half the xyz size to checkpoint center, 2. rotate according to rotation quaternion
                corner = Rotation.from_quat(rotation).apply(
                    np.array([0,
                              y * checkpoint_size[1] / 2,
                              z * checkpoint_size[2] / 2])).reshape((1, -1))

                corners = np.vstack((corners,
                                     corner))
        # plot current corner
        ax.plot(checkpoint_center[0] + corners[:, 0],
                checkpoint_center[1] + corners[:, 1],
                checkpoint_center[2] + corners[:, 2],
                color=color, linewidth=width)
    return ax


def track_from_logfile(
        filepath: str,
        ) -> pd.DataFrame:
    """
    Load a track from logfile.
    """
    track = pd.read_csv(filepath)
    ndict = {
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
    }
    track = track.rename(columns=ndict)
    track = track[list(ndict.values())]
    return track


def plot_trajectory_with_gates_3d(
        trajectory: pd.DataFrame,
        track: pd.DataFrame=None,
        sampling_rate: float=20.,
        xlims: tuple=(-15, 19),
        ylims: tuple=(-17, 17),
        zlims: tuple=(-8, 8),
        view: tuple=(45, 270),
        fig_size: tuple=(20, 10),
        ax: plt.axis=None,
        ) -> plt.axis:
    """
    Make a 3D trajectory plot with gates
    """
    if ax==None:
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1,projection='3d')
    if trajectory.shape[0] > 1:
        trajectory_sampling_rate = 1 / np.nanmedian(np.diff(trajectory.t.values))
        step_size = int(trajectory_sampling_rate / sampling_rate)
        indices = np.arange(0, trajectory.shape[0], step_size)
        trajectory = trajectory.iloc[indices, :]
        # Plot reference, track, and format figure.
        ax = plot_trajectory(
            trajectory.px.values,
            trajectory.py.values,
            trajectory.pz.values,
            trajectory.qx.values,
            trajectory.qy.values,
            trajectory.qz.values,
            trajectory.qw.values,
            axis_length=2,
            c='k',
            ax=ax,
        )
    ax = plot_gates_3d(
        track=track,
        ax=ax,
        color='k',
        width=4,
    )
    ax = format_trajectory_figure(
        ax=ax,
        xlims=xlims,
        ylims=ylims,
        zlims=zlims,
        xlabel='px [m]',
        ylabel='py [m]',
        zlabel='pz [m]',
        title='',
    )
    plt.axis('off')
    plt.grid(b=None)
    ax.view_init(elev=view[0],
                 azim=view[1])
    plt.gcf().set_size_inches(fig_size[0],
                              fig_size[1])
    return ax


def signed_horizontal_angle(
        reference_vector: np.array,
        target_vector: np.array,
        ) -> np.array:
    """
    Returns the signed horizontal angle between a given reference vector (
    v1) and another vector.
    """
    reference_vector = reference_vector[:, [0, 1]]
    target_vector = target_vector[:, [0, 1]]
    #normalize to unit vector length
    norm_v1 = reference_vector / np.linalg.norm(reference_vector,
                                                axis=1).reshape((-1, 1))
    norm_v2 = target_vector / np.linalg.norm(target_vector,
                                             axis=1).reshape((-1, 1))
    #compute the angle between the two vectors
    angle = np.array([np.arctan2(norm_v2[i, 1], norm_v2[i, 0]) - np.arctan2(norm_v1[i, 1], norm_v1[i, 0])
                      for i in range(norm_v1.shape[0])])

    for i in range(angle.shape[0]):
        #fix the angles above 180 degress
        if np.abs(angle[i]) > np.pi:
            if angle[i] < 0.:
                angle[i] = (2 * np.pi - np.abs(angle[i]))
            else:
                angle[i] = -(2 * np.pi - np.abs(angle[i]))
        #flip the sign
        if (np.abs(angle[i]) > 0.):
            angle[i] = -angle[i]
    return angle


def import_log(
        filepath,
        to_override=False,
        to_plot_traj_3d=False,
        to_plot_state=False,
        collider_dict={'gate-wall': ['gate', 'wall'],
                       'wall': ['wall']},
        collider_names=['gate-wall','wall'],
        track_filepaths={'flat': './tracks/flat.csv',
                         'wave': './tracks/wave.csv'},
        ) -> None:
    """Processes individual flight data logs: loading raw csv logs,
    track information, converts to standard format, computes gate passing
    events, plots drone states etc."""
    if isinstance(filepath,str):
        filepath=Path(filepath)
    print('..processing {}'.format(filepath))
    reference_filepath = filepath.parent/'original.csv'
    # Make output folder
    f = Path(filepath.as_posix()
                 .replace('.csv', '/')
                 .replace('/logs/', '/process/')
                 .replace('trajectory_', ''))
    if not f.exists():
        f.mkdir(parents=True,exist_ok=True)
    # Copy nw/mpc flown trajectory to output folder.
    if not (f/'trajectory.csv').exists():
        trajectory = trajectory_from_logfile(filepath=filepath)
        trajectory.to_csv(f/'trajectory.csv',index=False)
    # Copy reference trajectory to output folder.
    if not (f/'reference.csv').exists():
        reference = trajectory_from_logfile(filepath=reference_filepath)
        reference.to_csv(f/'reference.csv',index=False)
    # Copy track csv to output folder.
    if not (f/'track.csv').exists():
        if filepath.as_posix().find('wave') > -1:
            track_filepath = track_filepaths['wave']
        else:
            track_filepath = track_filepaths['flat']
        track = track_from_logfile(filepath=track_filepath)
        # Make some adjustments
        track['pz'] += 0.35  # shift gates up in flightmare
        track['dx'] = 0.
        track['dy'] = 3  # in the middle of inner diameter 2.5 and outer 3.0
        track['dz'] = 3  # in the middle of inner diameter 2.5 and outer 3.0
        track.to_csv(f/'track.csv',index=False)
    # Save gate pass and collision events to output folder.
    if not (f/'events.csv').exists():
        E = get_pass_collision_events(
            filepath_trajectory=f/'trajectory.csv',
            filepath_track=f/'track.csv')
        E.to_csv(f/'events.csv', index=False)
    # Loop across collider conditions
    for collider_name in collider_names:
        curr_colliders = collider_dict[collider_name]
        curr_feature_filename = 'features_{}.csv'.format(collider_name)
        # Save performance features to output folder
        if ((not (f/curr_feature_filename).exists()) or to_override):
            P = extract_performance_features(
                filepath_trajectory=f/'trajectory.csv',
                filepath_reference=f/'reference.csv',
                filepath_events=f/'events.csv',
                colliders=curr_colliders)
            P.to_csv(f/curr_feature_filename, index=False)
        # Save trajectory plot to output folder
        # todo: debug here, multiprocessing breaks when plotting
        if to_plot_traj_3d:
            track = pd.read_csv(f/'track.csv')
            trajectory = pd.read_csv(f/'trajectory.csv')
            features = pd.read_csv(f/curr_feature_filename)
            for label in ['', 'valid_']:
                for view, xlims, ylims, zlims in [
                    [(45, 270), (-15, 19), (-17, 17), (-8, 8)],
                    # [(0, 270), (-15, 19), (-17, 17), (-12, 12)],
                    # [(0, 180), (-15, 19), (-17, 17), (-12, 12)],
                    # [(90, 270), (-15, 19), (-15, 15), (-12, 12)],
                ]:
                    outpath = (f/'{}trajectory-with-gates_{}_{}x{}.jpg'
                        .format(label,collider_name,'%03d' % view[0], '%03d'
                        % view[1]))
                    if not outpath.exists():
                        if label == 'valid_':
                            ind = ((trajectory['t'].values >=
                                    features['t_start'].iloc[0]) &
                                   (trajectory['t'].values <=
                                    features['t_end'].iloc[0]))
                        else:
                            ind = np.array([True for i in range(
                                trajectory.shape[0])])
                        # todo: debug here, multiprocessing breaks when plotting
                        ax = plot_trajectory_with_gates_3d(
                            trajectory=trajectory.iloc[ind, :],
                            track=track,
                            view=view,
                            xlims=xlims,
                            ylims=ylims,
                            zlims=zlims,
                        )
                        ax.set_title(outpath)
                        plt.savefig(outpath)
                        plt.close()
                        ax = None

        # Plot the drone state
        # todo: debug here, multiprocessing breaks when plotting
        if to_plot_state:
            curr_state_filename = 'state_{}.jpg'.format(collider_name)
            if not (f/curr_state_filename).exists():
                # todo: debug here, multiprocessing breaks when plotting
                plot_state(
                    filepath_trajectory=f/'trajectory.csv',
                    filepath_reference=f/'reference.csv',
                    filepath_features=f/curr_feature_filename,
                )
                plt.savefig(f/curr_state_filename)
                plt.close(plt.gcf())


def confidence_interval(
        x,
        axis=0
        ):
    """Computes the 95% confidence interval of the standard error of means."""
    m = np.nanmean(x,axis=axis)
    s = np.nanstd(x,axis=axis,ddof=1) / np.sqrt(x.shape[axis])
    cl = m - 1.96 * s
    cu = m + 1.96 * s
    if axis==0:
        return np.vstack((cl,cu))
    else:
        return np.vstack((cl,cu)).T


def get_performance(
        collider_name,
        models,
    ) -> pd.DataFrame:
    """"Collects flight performance metrics across different flight datasets."""
    curr_feature_filename = 'features_{}.csv'.format(collider_name)
    performance = pd.DataFrame([])
    for model in models:
        filepaths = []
        for w in os.walk('./process/'+model+'/'):
            for f in w[2]:
                if f==curr_feature_filename:
                    filepaths.append(os.path.join(w[0], f))
        for filepath in filepaths:
            print('..collecting performance: {}'.format(filepath))
            df =  pd.read_csv(filepath)
            # Get model and run information from filepath
            strings = (
                df['filepath'].iloc[0]
                    .split('/process/')[-1]
                    .split('/trajectory.csv')[0]
                    .split('/')
            )
            if len(strings) == 2:
                strings.insert(1, 's016_r05_flat_li01_buffer20')
            # Load the yaml file
            yamlpath = None
            config = None
            yamlcount = 0
            for w in os.walk('./logs/'+model+'/'):
                for f in w[2]:
                    if f.find('.yaml')>-1:
                        yamlpath = os.path.join(w[0], f)
                        yamlcount += 1
            if yamlpath is not None:
                with open(yamlpath, 'r') as stream:
                    try:
                        config = yaml.safe_load(stream)
                    except yaml.YAMLError as exc:
                        print(exc)
                        config = None
            # Make Data dictionnairy for the output
            ddict = {}
            ddict['model_name'] = strings[0]
            ddict['has_dda'] = int(strings[0].find('dda') > -1)
            if ((strings[2].find('mpc_nw_act') > -1) &
                (filepath.find('mpc_nw_act') > -1)):
                ddict['has_network_used'] = 0
            else:
                ddict['has_network_used'] = 1

            ddict.setdefault('has_yaml', 0)
            ddict.setdefault('has_ref', 0)
            if config is not None:
                ddict['has_yaml'] = 1
                if 'no_ref' in config['train']:
                    ddict['has_ref'] = int(config['train']['no_ref'] == False)
                # Which drone state inputs were used.
                ddict['has_state_q'] = 0
                ddict['has_state_v'] = 0
                ddict['has_state_w'] = 0
                if 'use_imu' in config['train']:
                    if config['train']['use_imu'] == True:
                        ddict['has_state_q'] = 1
                        ddict['has_state_v'] = 1
                        ddict['has_state_w'] = 1
                        if 'imu_no_rot' in config['train']:
                            if config['train']['imu_no_rot'] == True:
                                ddict['has_state_q'] = 0
                        if 'imu_no_vels' in config['train']:
                            if config['train']['imu_no_vels'] == True:
                                ddict['has_state_v'] = 0
                                ddict['has_state_w'] = 0
                # Whether image features were used.
                ddict.setdefault('has_img', 0)
                if 'use_images' in config['train']:
                    ddict['has_img']=int(config['train']['use_images'])
                else:
                    if strings[0].find('_img')>-1:
                        ddict['has_img']=1
                # Whether feature tracks were used.
                ddict.setdefault('has_fts', 0)
                if 'use_fts_tracks' in config['train']:
                    ddict['has_fts'] = int(
                        config['train']['use_fts_tracks'])
                # Whether encoder features were used.
                ddict.setdefault('has_encfts', 0)
                if 'attention_fts_type' in config['train']:
                    if config['train']['attention_fts_type'] == \
                            'encoder_fts':
                        ddict['has_encfts'] = 1
                # Whether decoder features were used.
                ddict.setdefault('has_decfts', 0)
                if 'attention_fts_type' in config['train']:
                    if config['train']['attention_fts_type'] == \
                            'decoder_fts':
                        ddict['has_deccfts'] = 1
                # Whether gaze tracks used.
                ddict.setdefault('has_gztr', 0)
                if 'attention_fts_type' in config['train']:
                    if config['train']['attention_fts_type'] == \
                            'gaze_tracks':
                        ddict['has_gztr'] = 1
                # Whether attention branching was used
                ddict.setdefault('has_attbr', 0)
                if 'attention_branching' in config['train']:
                    if config['train']['attention_branching'] == True:
                        ddict['has_attbr'] = 1
                # Size of the time buffer used.
                ddict.setdefault('buffer', 0)
                if 'start_buffer' in config['simulation']:
                    ddict['buffer'] = config['simulation']['start_buffer']
            # Update the buffer time.
            ddict['buffer'] = float(strings[1].split('buffer')[-1]) / 10
            ddict['subject'] = int(
                strings[1].split('_')[0].split('-')[2].split('s')[-1])
            # Whether data is training or testing data.
            ddict.setdefault('has_train',0)
            ddict['has_train']=int(strings[1].split('_')[0].split(
                '-')[3]=='train')
            ddict.setdefault('has_test', 0)
            ddict['has_test']=int(strings[1].split('_')[0].split(
                '-')[3]=='test')
            # Dataset information
            ddict['run'] = int(strings[1].split('_')[1].replace('r',''))
            ddict['track'] = strings[1].split('_')[0].split('-')[1]
            li_string = strings[1].split('_')[2].replace('li','')
            if li_string.find('-')>-1:
                ddict['li'] = int(li_string.split('-')[0])
                ddict['num_laps'] = (int(li_string.split('-')[-1]) -
                                     int(li_string.split('-')[0]) + 1)
            else:
                ddict['li'] = int(li_string)
                ddict['num_laps'] = 1
            if ddict['has_dda'] == 0:
                if strings[2] == 'reference_mpc':
                    ddict['mt'] = -1
                    ddict['st'] = 0
                    ddict['repetition'] = 0
                else:
                    ddict['mt'] = -1
                    ddict['st'] = int(strings[2].split('_')[1].split('switch-')[-1])
                    ddict['repetition'] = int(strings[2].split('_')[-1])
            else:
                if strings[0].find('dda_offline')>-1:
                    ddict['mt'] = -1
                    ddict['st'] = int(strings[2].split('_')[1].split('st-')[-1])
                    ddict['repetition'] = int(strings[2].split('_')[-1])
                elif strings[2].find('mpc_eval_nw')>-1:
                    ddict['mt'] = -1
                    ddict['st'] = -1
                    ddict['repetition'] = 0
                elif strings[2].find('mpc_nw_act')>-1:
                    ddict['mt'] = -1
                    ddict['st'] = -1
                    ddict['repetition'] = 0
                else:
                    ddict['mt'] = int(strings[2].split('_')[1].split('mt-')[-1])
                    ddict['st'] = int(strings[2].split('_')[2].split('st-')[-1])
                    ddict['repetition'] = int(strings[2].split('_')[-1])
            # Add data dictionnairy as output row
            for k in sorted(ddict):
                df[k] = ddict[k]
            performance = performance.append(df)
    return performance


def make_summary_table(
        collider_name,
        curr_path,
        performance,
        online_name,
        trajectory_name
        ):
    print('----------------')
    print(online_name, trajectory_name)
    print('----------------')

    # Subject dictionnairy
    run_dict = None
    exclude_run_dict = None
    if trajectory_name == 'reference':
        run_dict = {
            'track': 'flat',
            'subject': 16,
            'run': 5,
            'li': 1,
            'num_laps': 1,
        }
    elif trajectory_name == 'other-laps':
        run_dict = {
            'track': 'flat',
            'num_laps': 1,
        }
        exclude_run_dict = {
            'track': 'flat',
            'subject': 16,
            'run': 5,
            'li': 1,
            'num_laps': 1,
        }
    elif trajectory_name == 'other-track':
        run_dict = {
            'track': 'wave',
            'num_laps': 1,
        }
    elif trajectory_name == 'multi-laps':
        run_dict = {
            'track': 'flat',
        }
        exclude_run_dict = {
            'num_laps': 1,
        }

    # Network general dictionnairy
    if online_name == 'online':
        network_dict = {
            'has_dda': 1,
            'has_network_used': 1,
        }
    else:
        network_dict = {
            'has_dda': 1,
            'has_network_used': 0,
        }

    # Model dictionnairy
    model_dicts = [
        {
            'name': 'Ref + RVW (Baseline)',
            'specs': {
                'has_ref': 1,
                'has_state_q': 1,
                'has_state_v': 1,
                'has_state_w': 1,
                'has_fts': 0,
                'has_decfts': 0,
                'has_attbr': 0,
                'has_gztr': 0,
            },
        },
        {
            'name': 'Ref + RVW + Fts',
            'specs': {
                'has_ref': 1,
                'has_state_q': 1,
                'has_state_v': 1,
                'has_state_w': 1,
                'has_fts': 1,
                'has_decfts': 0,
                'has_attbr': 0,
                'has_gztr': 0,
            },
        }, {
            'name': 'Ref + RVW + AIn',
            'specs': {
                'has_ref': 1,
                'has_state_q': 1,
                'has_state_v': 1,
                'has_state_w': 1,
                'has_fts': 0,
                'has_decfts': 1,
                'has_attbr': 0,
                'has_gztr': 0,
            },
        }, {
            'name': 'Ref + RVW + Abr',
            'specs': {
                'has_ref': 1,
                'has_state_q': 1,
                'has_state_v': 1,
                'has_state_w': 1,
                'has_fts': 0,
                'has_decfts': 0,
                'has_attbr': 1,
                'has_gztr': 0,
            },
        }, {
            'name': 'Ref (Baseline)',
            'specs': {
                'has_ref': 1,
                'has_state_q': 0,
                'has_state_v': 0,
                'has_state_w': 0,
                'has_fts': 0,
                'has_decfts': 0,
                'has_attbr': 0,
                'has_gztr': 0,
            },
        }, {
            'name': 'Ref + Fts',
            'specs': {
                'has_ref': 1,
                'has_state_q': 0,
                'has_state_v': 0,
                'has_state_w': 0,
                'has_fts': 1,
                'has_decfts': 0,
                'has_attbr': 0,
                'has_gztr': 0,
            },
        }, {
            'name': 'Ref + AIn',
            'specs': {
                'has_ref': 1,
                'has_state_q': 0,
                'has_state_v': 0,
                'has_state_w': 0,
                'has_fts': 0,
                'has_decfts': 1,
                'has_attbr': 0,
                'has_gztr': 0,
            },
        }, {
            'name': 'Ref + ABr',
            'specs': {
                'has_ref': 1,
                'has_state_q': 0,
                'has_state_v': 0,
                'has_state_w': 0,
                'has_fts': 0,
                'has_decfts': 0,
                'has_attbr': 1,
                'has_gztr': 0,
            },
        }, {
            'name': 'RVW (Baseline)',
            'specs': {
                'has_ref': 0,
                'has_state_q': 1,
                'has_state_v': 1,
                'has_state_w': 1,
                'has_fts': 0,
                'has_decfts': 0,
                'has_attbr': 0,
                'has_gztr': 0,
            },
        }, {
            'name': 'RVW + Fts',
            'specs': {
                'has_ref': 0,
                'has_state_q': 1,
                'has_state_v': 1,
                'has_state_w': 1,
                'has_fts': 1,
                'has_decfts': 0,
                'has_attbr': 0,
                'has_gztr': 0,
            },
        }, {
            'name': 'RVW + AIn',
            'specs': {
                'has_ref': 0,
                'has_state_q': 1,
                'has_state_v': 1,
                'has_state_w': 1,
                'has_fts': 0,
                'has_decfts': 1,
                'has_attbr': 0,
                'has_gztr': 0,
            },
        }, {
            'name': 'RVW + ABr',
            'specs': {
                'has_ref': 0,
                'has_state_q': 1,
                'has_state_v': 1,
                'has_state_w': 1,
                'has_fts': 0,
                'has_decfts': 0,
                'has_attbr': 1,
                'has_gztr': 0,
            },
        },
    ]

    # Feature dictionnairy
    if online_name == 'online':
        feature_dict = {
            'Flight Time [s]': {
                'varname': 'flight_time',
                'track': '',
                'first_line': 'mean',
                'second_line': 'std',
                'precision': 2
            },
            'Travel Distance [m]': {
                'varname': 'travel_distance',
                'track': '',
                'first_line': 'mean',
                'second_line': 'std',
                'precision': 2
            },
            'Mean Error [m]': {
                'varname': 'median_path_deviation',
                'track': '',
                'first_line': 'mean',
                'second_line': 'std',
                'precision': 2
            },
            'Gates Passed': {
                'varname': 'num_gates_passed',
                'track': '',
                'first_line': 'mean',
                'second_line': 'std',
                'precision': 2
            },
            '% Collision': {
                'varname': 'num_collisions',
                'track': '',
                'first_line': 'percent',
                'second_line': '',
                'precision': 0
            },
        }
    else:
        feature_dict = {
            'Throttle MSE': {
                'varname': 'throttle_error_mse-median',
                'track': '',
                'first_line': 'mean',
                'second_line': '',
                'precision': 3
            },
            'Throttle L1': {
                'varname': 'throttle_error_l1-median',
                'track': '',
                'first_line': 'mean',
                'second_line': '',
                'precision': 3
            },
            'Roll MSE': {
                'varname': 'roll_error_mse-median',
                'track': '',
                'first_line': 'mean',
                'second_line': '',
                'precision': 3
            },
            'Roll L1': {
                'varname': 'roll_error_l1-median',
                'track': '',
                'first_line': 'mean',
                'second_line': '',
                'precision': 3
            },
            'Pitch MSE': {
                'varname': 'pitch_error_mse-median',
                'track': '',
                'first_line': 'mean',
                'second_line': '',
                'precision': 3
            },
            'Pitch L1': {
                'varname': 'pitch_error_l1-median',
                'track': '',
                'first_line': 'mean',
                'second_line': '',
                'precision': 3
            },
            'Yaw MSE': {
                'varname': 'yaw_error_mse-median',
                'track': '',
                'first_line': 'mean',
                'second_line': '',
                'precision': 3
            },
            'Yaw L1': {
                'varname': 'yaw_error_l1-median',
                'track': '',
                'first_line': 'mean',
                'second_line': '',
                'precision': 3
            },
        }

    # Make a table
    if (run_dict is not None) and (model_dicts is not None):
        table = pd.DataFrame([])
        for mdict in model_dicts:

            # add subject dictionnairy to the model dictionnairy
            mdict['specs'] = {**mdict['specs'],
                              **network_dict,
                              **run_dict}

            ddict = {}
            ddict['Model'] = [mdict['name'], '']

            # Select performance data
            # Runs to include
            ind = np.array([True for i in range(performance.shape[0])])
            for k, v in mdict['specs'].items():
                ind = ind & (performance[k] == v)
            # Runs to exclude
            if exclude_run_dict is not None:
                ind_exclude = np.array([True for i in range(
                    performance.shape[0])])
                for k, v in exclude_run_dict.items():
                    ind_exclude = ind_exclude & (performance[k] == v)
                # Combine run selection
                ind = (ind == True) & (ind_exclude == False)
            # Select current runs
            curr_performance = performance.copy().loc[ind, :]

            ddict['Num Runs'] = [str(curr_performance.shape[0]), '']

            print(mdict['name'], ':', curr_performance.shape[0])
            # print(curr_performance)

            if curr_performance.shape[0] > 0:
                # Compute Average performances
                for outvar in feature_dict:
                    ddict.setdefault(outvar, [])
                    invar = feature_dict[outvar]['varname']
                    curr_vals = curr_performance[invar].values
                    # first line value
                    op1 = feature_dict[outvar]['first_line']
                    if op1 == 'mean':
                        val1 = np.nanmean(curr_vals)
                    elif op1 == 'percent':
                        val1 = 100 * np.mean(
                            (curr_vals > 0).astype(int))
                    else:
                        val1 = None
                    if val1 is None:
                        ddict[outvar].append('')
                    else:
                        ddict[outvar].append(
                            str(np.round(val1, feature_dict[outvar][
                                'precision'])))
                    # second line value
                    op2 = feature_dict[outvar]['second_line']
                    if op2 == 'std':
                        val1 = np.nanstd(curr_vals)
                    else:
                        val1 = None
                    if val1 is None:
                        ddict[outvar].append('')
                    else:
                        ddict[outvar].append(
                            '(' + str(
                                np.round(val1, feature_dict[outvar][
                                    'precision'])) + ')')

            for k in ddict:
                ddict[k] = [' '.join(ddict[k])]

            # Append two lines to the output table
            table = table.append(
                pd.DataFrame(ddict,
                             index=list(range(len(ddict['Model']))))
            )

        outpath = curr_path + '/{}/'.format(online_name)
        if not os.path.exists(outpath):
            make_path(outpath)

        outfilepath = outpath + 'latex_table_{}.csv'.format(
            trajectory_name)
        table.to_latex(outfilepath, index=False)


def get_subject_performance(
        base_path,
        to_plot_performance=True,
        to_plot_dist_successful_flight=True,
        collider_name='gate-wall',
        ):
    """ Collect performance data"""
    # Process new logfiles.
    outpath = base_path / 'analysis' / 'performance' / collider_name / \
              'subject_performance.csv'
    if not outpath.exists():
        filepaths = sorted((base_path / 'analysis' / 'process').rglob(
            '*/features_{}.csv'.format(collider_name)))
        data = pd.DataFrame([])
        for f in filepaths:
            df = pd.read_csv(f)
            df['model'] = f.parts[-4].split('_')[1]
            if f.parts[-3].find('trajectory') > -1:
                df['track'] = f.parts[-3].split('_')[0].split('-')[1]
                df['subject'] = int(
                    f.parts[-3].split('_')[0].split('-')[2].replace('s', ''))
                df['dataset'] = f.parts[-3].split('_')[0].split('-')[3]
            else:
                df['track'] = f.parts[-3].split('_')[2]
                df['subject'] = int(f.parts[-3].split('_')[0].replace('s', ''))
                df['dataset'] = 'test'
            if f.as_posix().find('mpc_nw_act')>-1:
                df['control']='mpc'
                df['mt'] = -1
                df['st'] = -1
            else:
                df['control']='nw'
                df['mt'] = int(f.parts[-2].split('_')[1].split('-')[1])
                df['st'] = int(f.parts[-2].split('_')[2].split('-')[1])
            df['trial'] = int(f.parts[-2].split('_')[3])
            data = data.append(df)
        if not outpath.parent.exists():
            outpath.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(outpath, index=False)
    # Extract performance from logfiles.
    inpath = base_path / 'analysis' / 'performance' / collider_name / 'subject_performance.csv'
    outpath = base_path / 'analysis' / 'performance' / collider_name / 'summary.csv'
    if (inpath.exists()) & (not outpath.exists()):
        data = pd.read_csv(inpath)
        ddict = {}
        for model in data['model'].unique():
            for track in data['track'].unique():
                for dataset in data['dataset'].unique():
                    print('--------------------------------')
                    print('model, track, dataset, subject, num_samples,'+
                              'num_coll_free, prop_coll_free')
                    print('--------------------------------')
                    for subject in data['subject'].unique():
                        ind = (
                                (data['model'].values == model) &
                                (data['track'].values == track) &
                                (data['dataset'].values == dataset) &
                                (data['subject'].values == subject) &
                                (data['control'].values == 'nw')
                        )
                        num_samples = np.sum(ind)
                        num_coll_free = np.sum(
                            data.loc[ind, 'num_collisions'].values == 0)
                        num_gates_passed = np.sum(
                            data.loc[ind, 'num_gates_passed'].values >= 10)
                        prop_coll_free = num_coll_free / num_samples
                        prop_gates_passed = num_gates_passed / num_samples
                        ddict.setdefault('model', [])
                        ddict['model'].append(model)
                        ddict.setdefault('track', [])
                        ddict['track'].append(track)
                        ddict.setdefault('dataset', [])
                        ddict['dataset'].append(dataset)
                        ddict.setdefault('subject', [])
                        ddict['subject'].append(subject)
                        ddict.setdefault('num_samples', [])
                        ddict['num_samples'].append(num_samples)
                        ddict.setdefault('num_coll_free', [])
                        ddict['num_coll_free'].append(num_coll_free)
                        ddict.setdefault('num_gates_passed', [])
                        ddict['num_gates_passed'].append(num_gates_passed)
                        ddict.setdefault('prop_coll_free', [])
                        ddict['prop_coll_free'].append(prop_coll_free)
                        ddict.setdefault('prop_gates_passed', [])
                        ddict['prop_gates_passed'].append(prop_gates_passed)
                        print(model, track, dataset, subject, num_samples,
                              num_coll_free, prop_coll_free)
        summary = pd.DataFrame(ddict)
        summary.to_csv(outpath, index=False)
    # Plot performance tables and figure.
    if to_plot_performance:
        inpath = base_path / 'analysis' / 'performance' / collider_name / 'subject_performance.csv'
        inpath2 = base_path / 'analysis' / 'performance' / collider_name / 'summary.csv'
        outpath = base_path / 'analysis' / 'performance' / collider_name / 'plots'
        if inpath2.exists():
            # Load performance data
            data = pd.read_csv(inpath)
            summary = pd.read_csv(inpath2)
            # Loop over different model configurations
            for model in data['model'].unique():
                for track in data['track'].unique():
                    for dataset in data['dataset'].unique():
                        # Determine if any data is available:
                        ind = ((data['model'].values == model) &
                               (data['track'].values == track) &
                               (data['dataset'].values == dataset))
                        if np.sum(ind) > 0:
                            # Make a figure that shows trajectories for all subjects
                            fig, axs = plt.subplots(5, 4)
                            fig.set_figwidth(18)
                            fig.set_figheight(10)
                            axs = axs.flatten()
                            i = 0
                            for subject in data['subject'].unique():
                                # Determine Success rate
                                ind = ((summary['model'].values == model) &
                                       (summary['track'].values == track) &
                                       (summary['dataset'].values == dataset) &
                                       (summary['subject'].values == subject))
                                is_success = True
                                fontweight = 'normal'
                                fontcolor = 'black'
                                frame_highlight = False
                                gates_passed_rate = ''
                                collision_free_rate = ''
                                if np.sum(ind) > 0:
                                    _num_samples = summary.loc[
                                        ind, 'num_samples'].iloc[0]
                                    _num_coll_free = summary.loc[
                                        ind, 'num_coll_free'].iloc[0]
                                    _num_gates_passed = summary.loc[
                                        ind, 'num_gates_passed'].iloc[0]
                                    u = \
                                        summary.loc[ind, 'prop_gates_passed'].iloc[
                                            0]
                                    v = summary.loc[ind, 'prop_coll_free'].iloc[
                                        0]
                                    if not np.isnan(v):
                                        gates_passed_rate = ' | G: {}/{} ({:.0f}%)'.format(
                                            _num_gates_passed, _num_samples,
                                            u * 100)
                                        collision_free_rate = ' | C: {}/{} ({:.0f}%)'.format(
                                            _num_coll_free, _num_samples,
                                            v * 100)
                                        if (u < 1) | (v < 1):
                                            fontweight = 'bold'
                                            frame_highlight = True
                                            is_success = False
                                            fontcolor = 'red'
                                # Plot trajectory
                                ind = (
                                        (data['model'].values == model) &
                                        (data['track'].values == track) &
                                        (data['dataset'].values == dataset) &
                                        (data['subject'].values == subject) &
                                        (data['trial'].values == 0)
                                )
                                if np.sum(ind) > 0:
                                    f = (Path(data.loc[ind, 'filepath'].iloc[0])
                                         .parent / 'trajectory-with-gates_gate-wall_045x270.jpg')
                                    im = cv2.imread(f.as_posix())
                                    # crop image borders
                                    im = im[270:-340, 250:-250, :]
                                    # add color frame (if not full success)
                                    if not is_success:
                                        im = cv2.copyMakeBorder(im, 20, 20, 20,
                                                                20,
                                                                cv2.BORDER_CONSTANT,
                                                                value=(
                                                                    255, 0, 0))
                                    axs[i].imshow(im)
                                axs[i].axis('off')
                                axs[i].set_title('s%03d' % subject +
                                                 gates_passed_rate +
                                                 collision_free_rate,
                                                 fontweight=fontweight,
                                                 color=fontcolor)
                                # raise the panel counter
                                i += 1
                            # remove axis from remaining panels
                            for i in range(i, axs.shape[0]):
                                axs[i].axis('off')
                            plt.tight_layout()
                            # make output directory
                            if not outpath.exists():
                                outpath.mkdir(parents=True, exist_ok=True)
                            # save the figure
                            op = (outpath / ('trajectories_{}_{}_{}.jpg'.format(
                                model, track, dataset)))
                            fig.savefig(op.as_posix())
                            plt.close(fig)
                            fig = None
                            axs = None
                            # Pring overall success to prompt
                            ind = ((summary['model'].values == model) &
                                   (summary['track'].values == track) &
                                   (summary['dataset'].values == dataset))
                            num_samples = np.nansum(
                                summary.loc[ind, 'num_samples'].values)
                            num_coll_free = np.nansum(summary.loc[ind,
                                                                  'num_coll_free'].values)
                            num_gates_passed = np.nansum(summary.loc[ind,
                                                                     'num_gates_passed'].values)
                            prop_coll_free = np.nan
                            prop_gates_passed = np.nan

                            if num_samples > 0:
                                prop_coll_free = num_coll_free / num_samples
                                prop_gates_passed = num_gates_passed / num_samples
                                print(
                                    'Success trajectories: {} {} {} | G: {}/{} ({:.0f}%) | C: {}/{} ({:.0f}%)'.format(
                                        model, track, dataset,
                                        num_gates_passed, num_samples,
                                        100 * prop_gates_passed,
                                        num_coll_free, num_samples,
                                        100 * prop_coll_free))
    # Plot proportion of successful laps for each of the tracks
    #todo: here split by network model
    if to_plot_dist_successful_flight:
        inpath = base_path / 'analysis' / 'performance' / collider_name / 'summary.csv'
        outpath = base_path / 'analysis' / 'performance' / collider_name / 'success_by_subject.csv'
        if inpath.exists():
            summary = pd.read_csv(inpath)
            ddict = {}
            for subject in summary['subject'].unique():
                # get laptime
                f = [v for v in sorted((base_path / 'analysis' / 'logs').rglob(
                    '*original.csv'))
                     if v.as_posix().find('s%03d' % subject) > -1]
                f = f[0]
                # print('..loading reference trajectory {}'.format(f))
                df = pd.read_csv(f)
                laptime = (df['time-since-start [s]'].iloc[-1] -
                           df['time-since-start [s]'].iloc[0])
                # get other performance features
                ind = ((summary['subject'].values == subject) &
                       (summary['dataset'].values == 'train'))
                num_samples = np.sum(summary.loc[ind, 'num_samples'].values)
                num_gates_passed = np.sum(
                    summary.loc[ind, 'num_gates_passed'].values)
                num_coll_free = np.sum(summary.loc[ind, 'num_coll_free'].values)
                prop_gates_passed = num_gates_passed / num_samples
                prop_coll_free = num_coll_free / num_samples
                ddict.setdefault('subject', [])
                ddict['subject'].append(subject)
                ddict.setdefault('laptime', [])
                ddict['laptime'].append(laptime)
                ddict.setdefault('num_samples', [])
                ddict['num_samples'].append(num_samples)
                ddict.setdefault('num_gates_passed', [])
                ddict['num_gates_passed'].append(num_gates_passed)
                ddict.setdefault('num_coll_free', [])
                ddict['num_coll_free'].append(num_coll_free)
                ddict.setdefault('prop_gates_passed', [])
                ddict['prop_gates_passed'].append(prop_gates_passed)
                ddict.setdefault('prop_coll_free', [])
                ddict['prop_coll_free'].append(prop_coll_free)
            df = pd.DataFrame(ddict)
            print(df)
            df.to_csv(outpath, index=False)

            plt.figure()
            plt.gcf().set_figwidth(15)
            plt.gcf().set_figheight(10)
            plt.subplot(2, 1, 1)
            plt.bar(df['subject'].values - 0.15,
                    df['prop_gates_passed'],
                    width=0.3,
                    label='Gates')
            plt.bar(df['subject'].values + 0.15,
                    df['prop_coll_free'],
                    width=0.3,
                    label='Collisions')
            plt.xticks(df['subject'].values)
            plt.xlabel('Subject')
            plt.ylabel('Proportion Successful Laps')
            plt.legend()
            plt.subplot(2, 1, 2)
            plt.bar(df['subject'].values,
                    df['laptime'],
                    width=0.5,
                    label='Lap Time')
            plt.plot([df['subject'].min() - 0.25, df['subject'].max() + 0.25],
                     np.ones((2,)) * np.nanmedian(df['laptime'].values),
                     '--r',
                     lw=3,
                     label='Median')
            plt.xticks(df['subject'].values)
            plt.xlabel('Subject')
            plt.ylabel('Lap Time [s]')
            plt.tight_layout()
            plt.savefig(outpath.as_posix().replace('.csv', '.jpg'))


def clean_performance_table(
        p
        ):
    """Clean performance table for subsequent analyses."""
    p['model']=[n.split('-')[0] for n in p['model'].values]
    ind=[True if n != 'attbr' else False for n in p['model'].values]
    p=p.loc[ind,:]
    return p


def get_average_performance(
        p
        ):
    """Collect a network performance summary, i.e. mean across 10 flights of the network per trajectory"""
    ddict={}
    for subject in p['subject'].unique():
        for dataset in p['dataset'].unique():
            for model in p['model'].unique():
                for track in p['track'].unique():
                    for control in p['control'].unique():
                        ind=(
                            (p['subject'].values==subject) &
                            (p['dataset'].values==dataset) &
                            (p['model'].values==model) &
                            (p['track'].values==track) &
                            (p['control'].values==control)
                            )
                        n='subject'
                        ddict.setdefault(n,[])
                        ddict[n].append(subject)
                        n='dataset'
                        ddict.setdefault(n, [])
                        ddict[n].append(dataset)
                        n = 'model'
                        ddict.setdefault(n, [])
                        ddict[n].append(model)
                        n = 'track'
                        ddict.setdefault(n, [])
                        ddict[n].append(track)
                        n = 'control'
                        ddict.setdefault(n, [])
                        ddict[n].append(control)
                        n = 'num_trials'
                        ddict.setdefault(n, [])
                        ddict[n].append(np.sum(ind))
                        for i in np.arange(0,11,1):
                            n = 'sr{}'.format(i)
                            ddict.setdefault(n, [])
                            value = np.mean((p.loc[ind,'num_gates_passed'].values>=i).astype(float))
                            ddict[n].append(value)
                        for n in ['travel_distance',
                           'median_path_deviation', 'iqr_path_deviation', 'num_gates_passed','num_collisions', 'network_used',
                           'pitch_error_l1', 'pitch_error_l1-median', 'pitch_error_mse',
                           'pitch_error_mse-median', 'roll_error_l1', 'roll_error_l1-median',
                           'roll_error_mse', 'roll_error_mse-median', 'throttle_error_l1',
                           'throttle_error_l1-median', 'throttle_error_mse',
                           'throttle_error_mse-median', 'yaw_error_l1', 'yaw_error_l1-median',
                           'yaw_error_mse', 'yaw_error_mse-median',]:
                            ddict.setdefault(n, [])
                            ddict[n].append(np.nanmean(p.loc[ind,n].values))
    r=pd.DataFrame(ddict)
    r=r.sort_values(by=['model','subject','dataset'])
    return r


def plot_success_rate_by_gate(r,
        dataset='test',
        control='nw',
        models=['img', 'ftstr', 'encfts'],
        names=['RGB Images','Feature Tracks', 'Attention Prediction'],
        colors=['k', 'b', 'r'],
        ) -> None:
    """Plot success rate of different networks as a function of gates passed."""
    plt.figure()
    plt.gcf().set_figwidth(6)
    plt.gcf().set_figheight(3)
    icolor=0
    iname=0
    for model in models:
        ind=(
            (r['dataset'].values==dataset) &
            (r['control'].values==control) &
            (r['model'].values==model)
        )
        c=colors[icolor]
        m=names[iname]
        x = np.arange(0, 11, 1)
        values=r.loc[ind,(['sr{}'.format(i) for i in x])].values
        ci=confidence_interval(values)
        y=np.mean(values,axis=0)
        plt.fill_between(x,ci[0,:],ci[1,:],color=c,alpha=0.1)
        plt.plot(x,y,'-o',color=c,label='{}'.format(m))
        icolor+=1
        iname+=1
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
               mode="expand", borderaxespad=0, ncol=3)
    plt.xticks(x)
    plt.yticks(np.arange(0,1.1,0.25))
    plt.xlabel('Gates Passed')
    plt.ylabel('Success Rate')
    plt.xlim((0.9,10.1))
    plt.ylim((0,1.025))
    plt.grid(axis='y')
    plt.tight_layout()


def get_command_prediction_table(r,
        dataset='test',
        control='mpc',
        models=['img', 'ftstr', 'encfts'],
        names=['RGB Images', 'Feature Tracks', 'Attention Prediction'],
        precision=2,
        ) -> None:
    """Make a table of command prediction resusults for different networks"""
    ddict={}
    imodel=0
    for model in models:
        ind=(
            (r['dataset'].values==dataset) &
            (r['control'].values==control) &
            (r['model'].values==model)
        )
        m=names[imodel]
        n='Model Name'
        ddict.setdefault(n,[])
        ddict[n].append(m)
        for cmd in ['throttle','roll','pitch', 'yaw']:
            for metric in ['mse','l1']:
                n='{}_{}'.format(cmd,metric)
                ddict.setdefault(n,[])
                values=r.loc[ind,'{}_error_{}'.format(cmd,metric)].values
                y=np.nanmean(values)
                ddict[n].append(np.round(y,precision))
        imodel+=1
    t=pd.DataFrame(ddict)
    t=t.set_index('Model Name')
    t=t.T
    t['Command'] = [n.split('_')[0].capitalize() for n in t.index]
    t['Metric'] = [n.split('_')[1].upper() for n in t.index]
    t=t.set_index(['Command','Metric']).T
    return t


def plot_reference_trajectory(
        base_path,
        track_dict
        ):
    """Plot reference trajectory for given tracks."""
    # Load track.
    track = pd.read_csv(track_dict['flat'])
    ndict = {
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
    }
    track = track.rename(columns=ndict)
    track = track[list(ndict.values())]
    track['pz'] += 0.35
    track['dx'] = 0.
    track['dy'] = 3
    track['dz'] = 3
    # Load reference and downsample to 20 Hz
    reference = trajectory_from_logfile(
        filepath=(base_path/'analysis'/'tracks'/'flat.csv').as_posix())
    sr = 1 / np.nanmedian(np.diff(reference.t.values))
    reference = reference.iloc[np.arange(0, reference.shape[0], int(sr / 20)), :]
    # Plot reference, track, and format figure.
    ax = plot_trajectory(
        reference.px.values,
        reference.py.values,
        reference.pz.values,
        reference.qx.values,
        reference.qy.values,
        reference.qz.values,
        reference.qw.values,
        axis_length=2,
        c='k',
    )
    ax = plot_gates_3d(
        track=track,
        ax=ax,
        color='k',
        width=4,
        )
    ax = format_trajectory_figure(
        ax=ax,
        xlims=(-15, 19),
        ylims=(-17, 17),
        zlims=(-8, 8),
        xlabel='px [m]',
        ylabel='py [m]',
        zlabel='pz [m]',
        title='',
        )

    plt.axis('off')
    plt.grid(b=None)
    ax.view_init(elev=45,
                 azim=270)
    plt.gcf().set_size_inches(20,10)

    plot_path = './plots/'
    if not os.path.exists(plot_path):
        make_path(plot_path)

    plt.savefig(plot_path + 'reference_3d.jpg')


def plot_reference_trajectory_with_decision(
        ):
    data_path = './branching_demo/'
    # Load track.
    track = pd.read_csv(data_path + 'flat.csv')
    ndict = {
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
    }
    track = track.rename(columns=ndict)
    track = track[list(ndict.values())]
    track['pz'] += 0.35
    track['dx'] = 0.
    track['dy'] = 3
    track['dz'] = 3
    # Load reference and downsample to 20 Hz
    reference = trajectory_from_logfile(
        filepath=data_path + 'trajectory_reference_original.csv')
    sr = 1 / np.nanmedian(np.diff(reference.t.values))
    reference = reference.iloc[
                np.arange(0, reference.shape[0], int(sr / 20)),
                :]
    # Plot reference, track, and format figure.
    ax = plot_trajectory(
        reference.px.values,
        reference.py.values,
        reference.pz.values,
        reference.qx.values,
        reference.qy.values,
        reference.qz.values,
        reference.qw.values,
        axis_length=2,
        c='b',
        axis_colors='b',
    )
    ax = plot_gates_3d(
        track=track,
        ax=ax,
        color='k',
        width=4,
    )
    ax = format_trajectory_figure(
        ax=ax,
        xlims=(-15, 19),
        ylims=(-17, 17),
        zlims=(-8, 8),
        xlabel='px [m]',
        ylabel='py [m]',
        zlabel='pz [m]',
        title='',
    )

    plt.axis('off')
    plt.grid(b=None)
    ax.view_init(elev=45,
                 azim=270)
    plt.gcf().set_size_inches(20, 10)

    plot_path = './plots/'
    if not os.path.exists(plot_path):
        make_path(plot_path)

    plt.savefig(plot_path + 'reference_flat_with_decision.jpg')

