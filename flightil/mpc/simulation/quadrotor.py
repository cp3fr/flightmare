import numpy as np
from scipy.spatial.transform import Rotation as R

from mpc.simulation.quad_index import *


class QuadRotor:

    def __init__(self, simulation_time_step):
        self.state_dim = 10
        self.action_dim = 4

        self._state = np.zeros(shape=self.state_dim)
        self._state[kQuatW] = 1.0

        self._actions = np.zeros(shape=self.action_dim)

        self._gravity = 9.81
        self._simulation_time_step = simulation_time_step
        self._arm_length = 0.3  # m
        self._mass = 1.0  # 3.2

        # Sampling range of the quadrotor's initial position
        # TODO: this would have to be around some (probably specified initial starting point)
        self._xyz_dist = np.array(
            [[-2.5, 2.5],  # x
             [0.0, 5.0],  # y
             [2.5, 4.0]]  # z
        )
        # Sampling range of the quadrotor's initial velocity
        # TODO: this can probably stay the same? but might want to start of with 0 velocity
        self._vxyz_dist = np.array(
            [[-1.0, 1.0],  # vx
             [-1.0, 1.0],  # vy
             [-1.0, 1.0]]  # vz
        )

        self._wxyz_dist = np.array(
            [[-np.pi / 12, np.pi / 12],
             [-np.pi / 12, np.pi / 12],
             [-np.pi / 12, np.pi / 12]]
        )

        # x, y, z, r, p, y, vx, vy, vz
        # TODO: probably remove, since these don't seem to be used
        self.obs_low = np.array([-10, -10, -10, -np.pi, -np.pi, -np.pi, -10, -10, -10])
        self.obs_high = np.array([10, 10, 10, np.pi, np.pi, np.pi, 10, 10, 10])

        self.reset()

    def reset(self):
        # TODO: might e.g. want to specify a specific state

        self._state = np.zeros(shape=self.state_dim)
        self._state[kQuatW] = 1.0

        # initialize position randomly
        self._state[kPosX] = np.random.uniform(
            low=self._xyz_dist[0, 0], high=self._xyz_dist[0, 1])
        self._state[kPosY] = np.random.uniform(
            low=self._xyz_dist[1, 0], high=self._xyz_dist[1, 1])
        self._state[kPosZ] = np.random.uniform(
            low=self._xyz_dist[2, 0], high=self._xyz_dist[2, 1])

        # initialize rotation randomly
        quad_quat0 = np.random.uniform(low=0.0, high=1, size=4)
        # angle = np.random.uniform(low=self._wxyz_dist[0, 0], high=self._wxyz_dist[0, 1])
        # quad_quat0 = np.array([np.cos(0.5 * angle), np.sin(0.5 * angle), 0.0, 0.0])
        # normalize the quaternion
        self._state[kQuatW:kQuatZ + 1] = quad_quat0 / np.linalg.norm(quad_quat0)

        # initialize velocity randomly
        self._state[kVelX] = np.random.uniform(
            low=self._vxyz_dist[0, 0], high=self._vxyz_dist[0, 1])
        self._state[kVelY] = np.random.uniform(
            low=self._vxyz_dist[1, 0], high=self._vxyz_dist[1, 1])
        self._state[kVelZ] = np.random.uniform(
            low=self._vxyz_dist[2, 0], high=self._vxyz_dist[2, 1])

        return self._state

    def run(self, action):
        """
        Apply the control command on the quadrotor and transits the system to the next state
        """

        refine_steps = 4
        refine_dt = self._simulation_time_step / refine_steps

        # Runge-Kutta 4th order integration
        state = self._state
        for i in range(refine_steps):
            k1 = refine_dt * self._f_quad_dynamics(state, action)
            k2 = refine_dt * self._f_quad_dynamics(state + 0.5 * k1, action)
            k3 = refine_dt * self._f_quad_dynamics(state + 0.5 * k2, action)
            k4 = refine_dt * self._f_quad_dynamics(state + k3, action)

            state = state + (k1 + 2.0 * (k2 + k3) + k4) / 6.0

        self._state = state
        # self._state[kQuatW:kQuatZ + 1] = self.get_quaternion()
        return self._state

    def _f_quad_dynamics(self, state, action):
        """
        System dynamics: ds = f(x, u)
        """
        thrust, wx, wy, wz = action
        thrust /= self._mass

        state_dot = np.zeros(shape=self.state_dim)

        state_dot[kPosX:kPosZ + 1] = state[kVelX:kVelZ + 1]

        qw, qx, qy, qz = self.get_quaternion()

        state_dot[kQuatW] = 0.5 * (-wx * qx - wy * qy - wz * qz)
        state_dot[kQuatX] = 0.5 * (wx * qw + wz * qy - wy * qz)
        state_dot[kQuatY] = 0.5 * (wy * qw - wz * qx + wx * qz)
        state_dot[kQuatZ] = 0.5 * (wz * qw + wy * qx - wx * qy)

        state_dot[kVelX] = 2 * (qw * qy + qx * qz) * thrust
        state_dot[kVelY] = 2 * (qy * qz - qw * qx) * thrust
        state_dot[kVelZ] = (qw * qw - qx * qx - qy * qy + qz * qz) * thrust - self._gravity

        return state_dot

    def set_state(self, state):
        """
        Set the vehicle's state
        """
        self._state = state

    def get_state(self):
        """
        Get the vehicle's state
        """
        return self._state

    def get_cartesian_state(self):
        """
        Get the Full state in Cartesian coordinates
        """
        cartesian_state = np.zeros(shape=9)
        cartesian_state[0:3] = self.get_position()
        cartesian_state[3:6] = self.get_euler()
        cartesian_state[6:9] = self.get_velocity()
        return cartesian_state

    def get_position(self):
        """
        Retrieve Position
        """
        return self._state[kPosX:kPosZ + 1]

    def get_velocity(self):
        """
        Retrieve Linear Velocity
        """
        return self._state[kVelX:kVelZ + 1]

    def get_quaternion(self):
        """
        Retrieve Quaternion
        """
        quat = self._state[kQuatW:kQuatZ + 1]
        quat = quat / np.linalg.norm(quat)
        return quat

    def get_euler(self):
        """
        Retrieve Euler Angles of the Vehicle
        """
        quat = self.get_quaternion()
        euler = self._quat_to_euler(quat)
        return euler

    def get_axes(self):
        """
        Get the 3 axes (x, y, z) in world frame (for visualization only)
        """
        # TODO: I'll leave this here for now, because it might help figuring out which way the camera should be pointing
        # axes in body frame
        b_x = np.array([self._arm_length, 0, 0])
        b_y = np.array([0, self._arm_length, 0])
        b_z = np.array([0, 0, -self._arm_length])

        # rotation matrix
        rot_matrix = R.from_quat(self.get_quaternion()).as_matrix()
        quad_center = self.get_position()

        # axes in body frame
        w_x = rot_matrix @ b_x + quad_center
        w_y = rot_matrix @ b_y + quad_center
        w_z = rot_matrix @ b_z + quad_center
        return [w_x, w_y, w_z]

    def get_motor_pos(self):
        """
        Get the 4 motor poses in world frame (for visualization only)
        """
        # TODO: see get_axes
        # motor position in body frame
        b_motor1 = np.array([np.sqrt(self._arm_length / 2), np.sqrt(self._arm_length / 2), 0])
        b_motor2 = np.array([-np.sqrt(self._arm_length / 2), np.sqrt(self._arm_length / 2), 0])
        b_motor3 = np.array([-np.sqrt(self._arm_length / 2), -np.sqrt(self._arm_length / 2), 0])
        b_motor4 = np.array([np.sqrt(self._arm_length / 2), -np.sqrt(self._arm_length / 2), 0])
        #
        rot_matrix = R.from_quat(self.get_quaternion()).as_matrix()
        quad_center = self.get_position()

        # motor position in world frame
        w_motor1 = rot_matrix @ b_motor1 + quad_center
        w_motor2 = rot_matrix @ b_motor2 + quad_center
        w_motor3 = rot_matrix @ b_motor3 + quad_center
        w_motor4 = rot_matrix @ b_motor4 + quad_center
        return [w_motor1, w_motor2, w_motor3, w_motor4]

    @staticmethod
    def _quat_to_euler(quat):
        """
        Convert Quaternion to Euler Angles
        """
        quat_w, quat_x, quat_y, quat_z = quat[0], quat[1], quat[2], quat[3]
        euler_x = np.arctan2(2 * quat_w * quat_x + 2 * quat_y * quat_z,
                             quat_w * quat_w - quat_x * quat_x - quat_y * quat_y + quat_z * quat_z)
        euler_y = -np.arcsin(2 * quat_x * quat_z - 2 * quat_w * quat_y)
        euler_z = np.arctan2(2 * quat_w * quat_z + 2 * quat_x * quat_y,
                             quat_w * quat_w + quat_x * quat_x - quat_y * quat_y - quat_z * quat_z)
        return [euler_x, euler_y, euler_z]
