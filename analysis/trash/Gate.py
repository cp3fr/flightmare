import numpy as np
from shapely.geometry import LineString
from scipy.spatial.transform import Rotation


class Gate(object):

    def __init__(self, df, dims=None, dtype='gazesim'):
        #set variable names according to the type of data
        if dtype=='liftoff':
            position_varnames = ['position_x [m]', 'position_y [m]', 'position_z [m]']
            rotation_varnames = [ 'rotation_x [quaternion]', 'rotation_y [quaternion]', 'rotation_z [quaternion]',
                                  'rotation_w [quaternion]']
            dimension_varnames = ['checkpoint-size_y [m]', 'checkpoint-size_z [m]']
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
        #==========================================================================
        #old way:
        # proto = np.array([[-hw, -hw, hw, hw, -hw],
        #                   [0., 0., 0., 0., 0.],   #assume surface, thus no thickness
        #                   [-hh, hh, hh, -hh, -hh]])
        # self._corners = ((Rotation.from_euler('z', [np.pi / 2]).apply(Rotation.from_quat(q).apply(proto.T)).T +
        #                   p.reshape(3, 1)).astype(float))
        #===========================================================================
        #assuming gates are oriented in the direction of flight: x=forward, y=left, z=up
        proto = np.array([[ 0.,  0.,  0.,  0.,  0.],  # assume surface, thus no thickness along x axis
                          [ hw, -hw, -hw,  hw,  hw],
                          [ hh,  hh, -hh, -hh,  hh]])
        self._corners = (Rotation.from_quat(q).apply(proto.T).T + p.reshape(3, 1)).astype(float)
        self._center = p
        self._rotation = q
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

    def intersect(self, p0, p1):
        '''
        p0 and p1 are the start and endpoints of a line in W frame
        '''
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

            # print(line_xy)
            # print(line_xz)
            # print(line_yz)
            #
            # print(self.xy.intersects(line_xy), [val for val in self.xy.intersection(line_xy).coords])
            # print(self.xz.intersects(line_xz), [val for val in self.xz.intersection(line_xz).coords])
            # print(self.yz.intersects(line_yz), [val for val in self.yz.intersection(line_yz).coords])

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
            #     print('Crossing detected: ', point_3d, point_2d)
            # else:
            #     print('No crossing')

        return point_2d, point_3d

        #     if self.xy.intersects(line_xy):
        #         xy = [val for val in self.xy.intersection(line_xy).coords]
        #         if len(xy) == 2:
        #             ind = np.argmin(np.array([np.linalg.norm(np.array(xy[0]) - p0[:2]),
        #                                       np.linalg.norm(np.array(xy[1]) - p0[:2])]))
        #         else:
        #             ind = 0
        #         p_xy = xy[ind]
        #         b = p1-p0
        #         print('==============')
        #         print(p_xy)
        #         print(p0)
        #         print(p1)
        #         print(b)
        #         b /= np.linalg.norm(b)
        #         f = (p_xy[0]-p0[0])/b[0]
        #         print(f)
        #         print('==============')
        #         _p = p0+f*b
        #         p_z = _p[2]
        #         if (p_z >= self._z[0]) & (p_z <= self._z[1]):
        #             point_3d = np.array([p_xy[0], p_xy[1], p_z])
        #             point_2d = self.point2d(point_3d)
        # return point_2d, point_3d

    def point2d(self, p):
        '''
        For AIRR square gates placed vertically
        x=right
        y=down
        origin at top left
        '''
        # print(p)
        # print(self._rotation)
        # _p = p - self._center
        # print(_p)
        # _p = Rotation.from_quat(self._rotation).apply(_p)
        # print('----------')
        # print(_p)
        # print('----------')
        # p0_xy = self._corners[:2, 0] #gate horizontal axis origin
        # p1_xy = self._corners[:2, 2] #gate horizontal axis endpoint
        # x = np.linalg.norm(p[:2]-p0_xy) / np.linalg.norm(p1_xy-p0_xy)
        # p0_z = self._corners[2, 0]  # gate vertical axis origin
        # p1_z = self._corners[2, 2]  # gate vertical axis endpoint
        # y = np.abs(p[2] - p0_z) / np.abs(p1_z - p0_z)
        # return np.array([x, y])
        return None