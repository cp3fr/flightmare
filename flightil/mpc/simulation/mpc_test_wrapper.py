import numpy as np

from flightgym import MPCTest_v0, RacingEnv


class MPCTestWrapper:

    def __init__(self, wave_track=False, env=None):
        if env is None:
            import os
            self.env = MPCTest_v0(os.path.join(os.getenv("FLIGHTMARE_PATH"),
                                               "flightlib/configs/quadrotor_env.yaml"), wave_track)
            # self.env = MPCTest_v0("/home/simon/flightmare/flightlib/configs/quadrotor_env.yaml", wave_track)
        else:
            self.env = env

        self.image_height = self.env.getImageHeight()
        self.image_width = self.env.getImageWidth()

        self.image = np.zeros((self.image_height * self.image_width * 3,), dtype=np.uint8)

    def _reshape_image(self):
        return np.reshape(self.image, (3, self.image_height, self.image_width)).transpose((1, 2, 0))

    def step(self, new_state):
        if len(new_state.shape) == 2:
            new_state = np.reshape(new_state, (-1, 1))
        new_state = new_state.astype(np.float32)
        self.env.step(new_state, self.image)
        return self._reshape_image()

    def is_colliding(self):
        self.env.getCollision()

    def connect_unity(self, pub_port=10253, sub_port=10254):
        self.env.connectUnity(pub_port, sub_port)

    def disconnect_unity(self):
        self.env.disconnectUnity()


class RacingEnvWrapper:

    def __init__(self, wave_track=False, env=None):
        if env is None:
            import os
            self.env = RacingEnv(os.path.join(os.getenv("FLIGHTMARE_PATH"),
                                              "flightlib/configs/quadrotor_env.yaml"), wave_track)
        else:
            self.env = env

        self.image_height = self.env.getImageHeight()
        self.image_width = self.env.getImageWidth()
        self.state_dim = self.env.getStateDim()

        self.image = np.zeros((self.image_height * self.image_width * 3,), dtype=np.uint8)
        self.state = np.zeros((self.state_dim,), dtype=np.float32)

    def _reshape_image(self):
        return np.reshape(self.image, (3, self.image_height, self.image_width)).transpose((1, 2, 0))

    def step(self, action):
        if len(action.shape) == 2:
            action = np.reshape(action, (-1, 1))
        action = action.astype(np.float32)
        success = self.env.step(action)
        return success

    def get_image(self):
        self.env.getImage(self.image)
        return self._reshape_image()

    def get_state(self):
        self.env.getState(self.state)
        return self.state.copy()

    def get_sim_time_step(self):
        return self.env.getSimTimeStep()

    def set_sim_time_step(self, sim_time_step):
        self.env.setSimTimeStep(float(sim_time_step))

    def set_reduced_state(self, reduced_state):
        if len(reduced_state.shape) == 2:
            reduced_state = np.reshape(reduced_state, (-1, 1))
        reduced_state = reduced_state.astype(np.float32)
        self.env.setReducedState(reduced_state)

    # def is_colliding(self):
    #     self.env.getCollision()

    def connect_unity(self, pub_port=10253, sub_port=10254):
        self.env.connectUnity(pub_port, sub_port)

    def disconnect_unity(self):
        self.env.disconnectUnity()
