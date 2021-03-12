import os
import sys
import re
import cv2
import csv
import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pathlib import Path
from shutil import copyfile
from ffprobe import FFProbe
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d.art3d import Line3D
from shapely.geometry import LineString
from skspatial.objects import Vector, Points, Line, Point, Plane
from skspatial.plotting import plot_3d
from scipy.stats import iqr

class Checkpoint(object):
    """
    Checkpoint object represented as 2D surface in 3D space with x-axis
    pointing in the direction of flight/desired passing direction.

    Contains useful methods for determining distance of points,
    and intersections with the plane.
    """

    def __init__(self, df, dims=None, dtype='liftoff'):
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
            dims = (df[dimension_varnames[0]] * dimension_scaling_factor, df[dimension_varnames[1]] * dimension_scaling_factor)
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

    def get_distance(self, p):
        return self.plane.distance_point(p)

    def get_signed_distance(self, p):
        return self.plane.distance_point_signed(p)

    def intersect(self, p0: np.ndarray([]), p1: np.ndarray([])) -> tuple():
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

    def point2d(self, p: np.ndarray([])) -> np.ndarray([]):
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

def detect_checkpoint_pass(
        t: np.array,
        px: np.array,
        py: np.array,
        pz: np.array,
        checkpoint: Checkpoint,
        distance_threshold: int=None
        ) -> np.array:
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

## DEMO ##


# Load checkpoints in correct order
track = pd.read_csv('track.csv')
checkpoints = [Checkpoint(track.iloc[i]) for i in range(track.shape[0])]

# Load some prerecorded trajectory
trajectory = pd.read_csv('trajectory.csv')
t = trajectory['t'].values
p = trajectory.loc[:, ('px', 'py', 'pz')].values

# Set length of trajectory length to consider for event detection
length = 200

# Simulate sampling-based event detection
events = []
distance = []
next_gate = []
j = 0
_checkpoint = checkpoints[j]
for sample in range(t.shape[0]):

    # current "gate vector" and its length
    _v = _checkpoint.center - p[sample, :]
    _l = np.linalg.norm(_v)

    # collect some output for visual inspection
    next_gate.append(j)
    distance.append(_l)

    # short history of recent drone positions
    i0 = sample - length
    if i0 < 0:
        i0 = 0
    i1 = sample + 1
    _t = t[i0:i1]
    _px = p[i0:i1, 0]
    _py = p[i0:i1, 1]
    _pz = p[i0:i1, 2]

    # check for gate passing event
    _e = detect_checkpoint_pass(_t, _px, _py, _pz, _checkpoint)

    # if gate passing, set next checkpoint
    if len(_e) > 0:
        print(sample, j, _l, _v)

        j += 1
        if j >= len(checkpoints):
            j = 0
        _checkpoint = checkpoints[j]

    # collect some output for visual inspection
    if len(_e) > 0:
        events.append(1)
    else:
        events.append(0)

# Visualize the ouput
plt.subplot(3,1,1)
plt.plot(t, distance)
plt.ylabel('Distance from\nnext gate [m]')
plt.subplot(3,1,2)
plt.plot(t, events)
plt.ylabel('Checkpoint\nevent detected')
plt.subplot(3,1,3)
plt.plot(t, next_gate)
plt.xlabel('Time [s]')
plt.ylabel('Next checkpoint\n ID')

plt.savefig('output.jpg')
plt.show()

