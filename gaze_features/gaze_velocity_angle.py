import os
import sys
import cv2

import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt


# Settings
to_compute = True
to_plot = True


if to_compute:

    # Load drone state and resample to 50 Hz
    drone = pd.read_csv('drone.csv')
    ndict = {
        'ts': 't',
        'PositionX': 'px',
        'PositionY': 'py',
        'PositionZ': 'pz',
        'rot_x_quat': 'qx',
        'rot_y_quat': 'qy',
        'rot_z_quat': 'qz',
        'rot_w_quat': 'qw',
        'VelocityX': 'vx',
        'VelocityY': 'vy',
        'VelocityZ': 'vz',
    }
    drone = drone.rename(columns=ndict)
    drone = drone[list(ndict.values())]
    drone = drone.iloc[np.arange(0, drone.shape[0]+1, 10), :]

    # Load gaze and resample to 50 Hz
    gaze = pd.read_csv('gaze_on_surface.csv')
    ndict = {
        'ts': 't',
        'norm_x_su': 'x',
        'norm_y_su': 'y',
    }
    gaze = gaze.rename(columns=ndict)
    gaze = gaze[list(ndict.values())]
    gaze.at[:, 'y'] = 1 - gaze['y'].values
    ddict = {}
    for n in gaze.columns:
        ddict[n] = np.interp(
            drone.t.values,
            gaze.t.values,
            gaze.loc[:, n].values)
    gaze = pd.DataFrame(ddict, index=list(range(ddict['t'].shape[0])))

    # Camera settings.
    cam_fov = 120  # degrees
    cam_angle = 30  # degrees
    cam_pos_B = np.array([0.2, 0., 0.1])  # meters
    cam_rot_B = Rotation.from_euler('y',
                                    -cam_angle,
                                    degrees=True).as_quat()
    cam_matrix = np.array([[0.41, 0., 0.5],
                           [0., 0.56, 0.5],
                           [0., 0., 1.]])
    dist_coefs = np.array([[0., 0., 0., 0.]])

    # Compute camera pose in World frame.
    def pose_drone2cam(
            drone_pos_W: np.array,
            drone_rot_W: np.array,
            cam_pos_B: np.array,
            cam_rot_B: np.array,
            ) -> tuple:
        """
        Returns camera pose in world frame from given quadrotor pose in world
        frame and camera pose in body frame.
        """
        p_cam = Rotation.from_quat(drone_rot_W).apply(np.tile(cam_pos_B, (drone_pos_W.shape[0], 1))) + drone_pos_W
        r_cam = (Rotation.from_quat(drone_rot_W) * Rotation.from_quat(
            np.tile(cam_rot_B, (drone_pos_W.shape[0], 1)))).as_quat()
        return p_cam, r_cam
    cam_pos_W, cam_rot_W = pose_drone2cam(
        drone_pos_W=drone.loc[:, ('px', 'py', 'pz')].values,
        drone_rot_W=drone.loc[:, ('qx', 'qy', 'qz', 'qw')].values,
        cam_pos_B=cam_pos_B,
        cam_rot_B=cam_rot_B,
    )

    # Make camera pose dataframe (50 Hz)
    camera = pd.DataFrame({
        't': drone['t'].values,
        'px': cam_pos_W[:, 0],
        'py': cam_pos_W[:, 1],
        'pz': cam_pos_W[:, 2],
        'qx': cam_rot_W[:, 0],
        'qy': cam_rot_W[:, 1],
        'qz': cam_rot_W[:, 2],
        'qw': cam_rot_W[:, 3],
    }, index=list(range(drone.shape[0])))

    # Compute gaze vector in 3D
    def cam2world(
            r_cam: np.array,
            p_2d: np.array,
            camera_matrix: np.array,
            ) -> np.array:
        """
        Returns a normalized 3D projection of a given 2D camera frame
        coordinate.
        """
        p = np.hstack((p_2d,
                       np.ones((p_2d.shape[0], 1))))
        x = (np.linalg.pinv(camera_matrix @ np.eye(3)) @ p.T).T
        # convert from Opencv (x=right, y=down, z=forward) to
        # World (x=forward, y=left, z=up) coordinate frame
        x = np.hstack((x[:, 2:], np.hstack((-x[:, 0:1], -x[:, 1:2]))))
        x = x / np.linalg.norm(x, axis=1).reshape(-1, 1)
        x = Rotation.from_quat(r_cam).apply(x)
        return x

    gaze_vector = cam2world(
            r_cam=camera.loc[:, ('qx', 'qy', 'qz', 'qw')].values,
            p_2d=gaze.loc[:, ('x', 'y')].values,
            camera_matrix=cam_matrix,
            )
    gaze['nvx'] = gaze_vector[:, 0]
    gaze['nvy'] = gaze_vector[:, 1]
    gaze['nvz'] = gaze_vector[:, 2]


    # Compute signed angle between gaze vector and velocity vector
    def angle_between(
            v1: np.array,
            v2: np.array,
            ) -> np.array:
        """
        Returns the signed horizontal angle between a given reference vector (
        v1) and another vector.
        """
        v1 = v1[:, [0, 1]]
        v2 = v2[:, [0, 1]]
        #normalize to unit vector length
        norm_v1 = v1 / np.linalg.norm(v1, axis=1).reshape((-1, 1))
        norm_v2 = v2 / np.linalg.norm(v2, axis=1).reshape((-1, 1))
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
    gaze['angle'] = angle_between(
        v1=drone.loc[:, ('vx', 'vy', 'vz')].values,
        v2=gaze.loc[:, ('nvx', 'nvy', 'nvz')].values,
        )

    outpath = './output/'
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    drone.to_csv(outpath + 'drone.csv', index=False)
    gaze.to_csv(outpath + 'gaze.csv', index=False)
    camera.to_csv(outpath + 'camera.csv', index=False)


if to_plot:
    gaze = pd.read_csv('./output/gaze.csv')
    #load some previously computed gaze angles for comparison
    control = pd.read_csv('control.csv')
    plt.plot(gaze.t, gaze.angle)
    plt.plot(control.ts,
             control.angle2d_horz_velocity_gaze)
    plt.xlabel('time [s]')
    plt.ylabel('signed horizontal angle [rad]')
    plt.legend(['computed', 'control'])
    plt.show()