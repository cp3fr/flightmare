import numpy as np

from flightgym import MPCTest_v0


class MPCTestWrapper:

    def __init__(self, wave_track=False, env=None):
        if env is None:
            self.env = MPCTest_v0("/home/simon/Git/flightmare/flightlib/configs/quadrotor_env.yaml", wave_track)
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

    def connect_unity(self, pub_port=10253, sub_port=10254):
        self.env.connectUnity(pub_port, sub_port)

    def disconnect_unity(self):
        self.env.disconnectUnity()
