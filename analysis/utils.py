import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pprint import pprint
from mpl_toolkits.mplot3d.art3d import Line3D
from shapely.geometry import LineString
from scipy.spatial.transform import Rotation
from shutil import copyfile
from skspatial.objects import Vector, Points, Line, Point, Plane
from skspatial.plotting import plot_3d
from scipy.stats import iqr


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
        else: #gazesim: default
            position_varnames = ['pos_x', 'pos_y', 'pos_z']
            rotation_varnames = ['rot_x_quat', 'rot_y_quat', 'rot_z_quat', 'rot_w_quat']
            dimension_varnames = ['dim_y', 'dim_z']
            dimension_scaling_factor = 2.5
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

    objWallCollider.append(Checkpoint(pd.DataFrame({'pos_x': [center[0]],
                                              'pos_y': [center[1]],
                                              'pos_z': [center[2] - dims[2]/2],
                                              'rot_x_quat': [_q[0]],
                                              'rot_y_quat': [_q[1]],
                                              'rot_z_quat': [_q[2]],
                                              'rot_w_quat': [_q[3]],
                                              'dim_x': [0],
                                              'dim_y': [dims[1]],
                                              'dim_z': [dims[0]]},
                                                   index=[0]).iloc[0],
                                      dims=(dims[1], dims[0]), dtype='gazesim'))

    _q = (Rotation.from_euler('y', [-np.pi/2]) * Rotation.from_quat(np.array([0, 0, 0, 1]))).as_quat().flatten()
    objWallCollider.append(Checkpoint(pd.DataFrame({'pos_x': center[0], 'pos_y': center[1], 'pos_z' : center[2] + dims[2] / 2,
                                            'rot_x_quat': _q[0], 'rot_y_quat':_q[1], 'rot_z_quat':_q[2], 'rot_w_quat':_q[3],
                                            'dim_x':0, 'dim_y':dims[1], 'dim_z':dims[0]}, index=[0]).iloc[0],
                                      dims=(dims[1], dims[0]), dtype='gazesim'))

    _q = np.array([0, 0, 0, 1])
    objWallCollider.append(Checkpoint(pd.DataFrame({'pos_x': center[0] + dims[0] / 2, 'pos_y': center[1], 'pos_z' : center[2],
                                            'rot_x_quat': _q[0], 'rot_y_quat':_q[1], 'rot_z_quat':_q[2], 'rot_w_quat':_q[3],
                                            'dim_x':0, 'dim_y':dims[1], 'dim_z':dims[2]}, index=[0]).iloc[0],
                                      dims=(dims[1], dims[2]), dtype='gazesim'))

    _q = (Rotation.from_euler('z', [np.pi]) * Rotation.from_quat(np.array([0, 0, 0, 1]))).as_quat().flatten()
    objWallCollider.append(Checkpoint(pd.DataFrame({'pos_x': center[0] - dims[0] / 2, 'pos_y': center[1], 'pos_z' : center[2],
                                            'rot_x_quat': _q[0], 'rot_y_quat':_q[1], 'rot_z_quat':_q[2], 'rot_w_quat':_q[3],
                                            'dim_x':0, 'dim_y':dims[1], 'dim_z':dims[2]}, index=[0]).iloc[0],
                                      dims=(dims[1], dims[2]), dtype='gazesim'))

    _q = (Rotation.from_euler('z', [np.pi/2]) * Rotation.from_quat(np.array([0, 0, 0, 1]))).as_quat().flatten()
    objWallCollider.append(Checkpoint(pd.DataFrame({'pos_x': center[0], 'pos_y': center[1] + dims[1] / 2, 'pos_z' : center[2],
                                            'rot_x_quat': _q[0], 'rot_y_quat':_q[1], 'rot_z_quat':_q[2], 'rot_w_quat':_q[3],
                                            'dim_x':0, 'dim_y':dims[1], 'dim_z':dims[2]}, index=[0]).iloc[0],
                                      dims=(dims[0], dims[2]), dtype='gazesim'))

    _q = (Rotation.from_euler('z', [-np.pi/2]) * Rotation.from_quat(np.array([0, 0, 0, 1]))).as_quat().flatten()
    objWallCollider.append(Checkpoint(pd.DataFrame({'pos_x': center[0], 'pos_y': center[1] - dims[1] / 2, 'pos_z' : center[2],
                                            'rot_x_quat': _q[0], 'rot_y_quat':_q[1], 'rot_z_quat':_q[2], 'rot_w_quat':_q[3],
                                            'dim_x':0, 'dim_y':dims[1], 'dim_z':dims[2]}, index=[0]).iloc[0],
                                      dims=(dims[0], dims[2]), dtype='gazesim'))
    return objWallCollider


def detect_checkpoint_pass(
        t: np.ndarray([]),
        px: np.ndarray([]),
        py: np.ndarray([]),
        pz: np.ndarray([]),
        checkpoint: Checkpoint,
        distance_threshold: int=None
        ) -> np.ndarray([]):
    """
    Return timestamps when drone passes a checkpoint from given drone
    timestamps (t), position (px, py, pz), and checkpoint object.

        t: timestamps in seconds,
        px, py, pz: drone position in x, y, z in meters
        checkpoint: Gate object, i.e. 2D surface in 3D space
        distance_threshold: distance threshold in meters for which to
            consider candidate sampling point for detecting gate interaction

        Update on 12.02.2021
        Checks if position data is within a distance threshold from the gate
        And for those data checks if the gate was passed
    """
    position = np.hstack((px.reshape(-1, 1),
                               np.hstack((py.reshape(-1, 1),
                                          pz.reshape(-1, 1)))))

    # Set distance threshold to 60% of the gate surface diagonale.
    if distance_threshold is None:
        distance_threshold = 0.6 * np.sqrt((checkpoint.width) ** 2
                                           + (checkpoint.height) ** 2)
    # Select candidate timestamps in three steps:
    # First, find all timestamps when quad is close to gate.
    gate_center = checkpoint.center.reshape((1, 3)).astype(float)
    distance_from_gate_center = np.linalg.norm(position - gate_center, axis=1)
    timestamps_near_gate = t[distance_from_gate_center < distance_threshold]
    # Second, cluster the timestamps that occur consecutively
    dt = np.nanmedian(np.diff(t))
    ind = np.diff(timestamps_near_gate) > (4*dt)
    if len(ind) == 0:
        return []
    ind1 = np.hstack((ind, True))
    ind0 = np.hstack((True, ind))
    timstamp_clusters = np.hstack((
        timestamps_near_gate[ind0].reshape(-1, 1),
        timestamps_near_gate[ind1].reshape(-1, 1)
        ))
    # Third, find gate passing events using signed distances from gate plane.
    event_timestamps = []
    for cluster in range(timstamp_clusters.shape[0]):
        start_time = timstamp_clusters[cluster, 0]
        end_time = timstamp_clusters[cluster, 1]
        ind = (t>=start_time) & (t<=end_time)
        curr_time = t[ind]
        curr_position = position[ind, :]
        curr_signed_distance = np.array([
            checkpoint.get_signed_distance(curr_position[i, :]) for i in range(
                curr_position.shape[0]
            )
        ])
        # Find transitions from negative to positive signed distance.
        #  where "negative distance" is behind the gate (negative x in gate
        #  frame) and "positive distance" is in front of the gate (positive x
        #  in gate frame.
        ind = ((curr_signed_distance <= 0) & (
                np.diff(np.hstack((curr_signed_distance,
                                   curr_signed_distance[-1])) > 0) == 1)
               )
        if np.sum(ind) > 0:
            event_timestamps.append(curr_time[ind][0])
    return event_timestamps


def make_path(
        path: str
        ) -> bool():
    """Make (nested) folders, if not already existent, from provided path."""
    path = path.replace('//', '/')
    if path[0] == '/':
        curr_path = '/'
    else:
        curr_path = ''
    folders = path.split('/')
    made_some_folders = False
    for folder in folders:
        if len(folder) > 0:
            curr_path += folder + '/'
            if os.path.isdir(curr_path) == False:
                os.mkdir(curr_path)
                made_some_folders = True
    return made_some_folders


def trajectory_from_logfile(
        filepath: str
        ) -> pd.DataFrame():
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
    # Load race track information.
    T = pd.read_csv(filepath_track)
    # Define checkpoints and colliders.
    gate_checkpoints = [
        Checkpoint(T.iloc[i], dims=gate_inner_dimensions) for i in range(T.shape[0])]
    gate_colliders = [
        Checkpoint(T.iloc[i], dims=gate_outer_dimensions) for i in range(T.shape[0])]
    wall_colliders = get_wall_colliders(dims=wall_collider_dimensions,
                                        center=wall_collider_center)
    # Load a trajectory
    D = pd.read_csv(filepath_trajectory)
    t = D['t'].values
    p = D.loc[:, ('px', 'py', 'pz')].values
    # Detect checkpoint passing and collision events.
    events = {}
    for key, objects in [('gate_pass', gate_checkpoints),
                         ('gate_collision', gate_colliders),
                         ('wall_collision', wall_colliders)
                        ]:
        for id in range(len(objects)):
            object = objects[id]
            for timestamp in object.detect_pass(t=t, p=p):
                if not ((key == 'gate_collision') and (
                        timestamp in events.keys())):
                    events[timestamp] = (key, id)
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
            for k in sorted(events)]},
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
    # Compute offline features
    ind = (D['t'].values >= 0) & (D['network_used'].values == 0)
    if np.sum(ind)>0:
        network_used = 0
        mpc_nw_dict = {}
        for n in ['throttle', 'roll', 'pitch', 'yaw']:
            if ('{}_mpc'.format(n) not in D.columns) or ('{}_nw'.format(n)
                    not in D.columns):
                diff_values = np.nan
            else:
                diff_values = (D.loc[ind, '{}_mpc'.format(n)].values -
                               D.loc[ind, '{}_nw'.format(n)].values)
            mpc_nw_dict[n] = {
                'l1': np.nanmean(np.abs(diff_values)),
                'mse': np.nanmean(np.power(diff_values, 2)),
                'l1-median': np.nanmedian(np.abs(diff_values)),
                'mse-median': np.nanmedian(np.power(diff_values, 2)),
            }
    else:
        network_used = 1
        mpc_nw_dict = {}
        for n in ['throttle', 'roll', 'pitch', 'yaw']:
            mpc_nw_dict[n] = {
                'l1': np.nan,
                'mse': np.nan,
                'l1-median': np.nan,
                'mse-median': np.nan,
            }
    # Determine start and end time
    t_trajectory_start = D['t'].iloc[0]
    t_trajectory_end = D['t'].iloc[-1]
    ind = D['network_used'].values == 1
    if np.sum(ind) > 0:
        t_network_start = D['t'].values[ind][0]
        t_network_end = D['t'].values[ind][-1]
        network_in_control = True
    else:
        t_network_start = None
        t_network_end = None
        network_in_control = False
    ind = E['is-collision'].values == 1
    if np.sum(ind) > 0:
        t_first_collision = E.loc[ind, 't'].values[0]
    else:
        t_first_collision = None
    if t_network_start is not None:
        t_start = t_network_start
    else:
        t_start = t_trajectory_start
    if t_first_collision is not None:
        t_end = t_first_collision
    else:
        if t_network_end is not None:
            t_end = t_network_end
        else:
            t_end = t_trajectory_end

    # For consistency, start of the lap is at start/finish gate 9
    if t_start < 0:
        t_start = 0
    if t_end<t_start:
        t_end=t_start
    # Get performance metrics within the start and end time window
    # ..number of passed gates
    ind = (E['t'].values >= t_start) & (E['t'].values <= t_end)
    num_gates_passed = np.sum(E.loc[ind, 'is-pass'].values)

    ind = ((E['is-collision'].values == 1) &
           (E['t']>=t_start) &
           (E['t']<=t_end))
    num_collisions = np.sum(ind)

    num_passes = {}
    for i in range(10):
        ind3 = E['object-id'].values == i
        num_passes[i] = np.sum(E.loc[ind & ind3, 'is-pass'].values)

    # Distance and Duration features (considering all time the network was in
    # control)
    if t_end > t_start:
        total_distance = np.sum(
            np.linalg.norm(
                np.diff(
                    D.loc[((D['t']>=t_start) & (D['t']<=t_end)),
                        ('px', 'py', 'pz')].values,
                    axis=0),
                axis=1))
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
        fig , axs = plt.subplots(2, 1)
        fig.set_figwidth(20)
        fig.set_figheight(15)
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
    axs[iax].legend(['trajectory px', 'trajectory py', 'trajectory pz',
                     'reference px', 'reference py', 'reference pz', ],
                    loc='upper right')
    return axs


def compare_trajectories_3d(
        reference_filepath: str,
        data_path: str=None,
        ) -> None:
    """
    Comparison of 3D flight trajectories from two given trajectory logfile
    paths, showing 3D poses.
    """
    # Plot reference, MPC, and network trajectories in 3D
    if data_path:
        ref = trajectory_from_logfile(
            filepath=reference_filepath)
        ref = ref.iloc[np.arange(0, ref.shape[0], 50), :]
        ax = plot_trajectory(
            ref.px.values,
            ref.py.values,
            ref.pz.values,
            c='k')
        for w in os.walk(data_path):
            for f in w[2]:
                if (f.find('trajectory.csv') != -1):
                    if f.find('mpc_eval_nw') != -1:
                        color = 'r'
                    else:
                        color = 'b'
                    filepath = os.path.join(w[0], f)
                    print(filepath)
                    df = trajectory_from_logfile(
                        filepath=filepath)
                    print(df.columns)
                    plot_trajectory(df.px, df.py, df.pz, c=color, ax=ax)
        ax = format_trajectory_figure(
            ax, xlims=(-30, 30), ylims=(-30, 30), zlims=(-30, 30), xlabel='px [m]',
            ylabel='py [m]', zlabel='pz [m]', title=data_path)


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
        ) -> plt.axis:
    """
    Make a 3D trajectory plot with gates
    """
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
        )
    else:
        ax=None
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

