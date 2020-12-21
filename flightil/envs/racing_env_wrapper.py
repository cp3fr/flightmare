import numpy as np

from flightgym import RacingEnv


class RacingEnvWrapper:

    def __init__(self):
        self.env = RacingEnv()

        # TODO: should probably get more parameters (e.g. min/max thrust) or allow setting more parameters
        #  (although the latter should be taken care of using YAML files I guess)

        # getting all of the important information
        self.action_dim = self.env.getActDim()
        self.observation_dim = self.env.getObsDim()

        self.image_height = self.env.getImageHeight()
        self.image_width = self.env.getImageWidth()

        self.simulation_time_step = self.env.getSimTimeStep()

        # creating the shared variables
        self.state_obs = np.zeros((self.observation_dim,), dtype=np.uint8)
        self.image_obs = np.zeros((self.image_height * self.image_width * 3,), dtype=np.uint8)

    def _reshape_image_obs(self):
        return np.reshape(self.image_obs, (3, self.image_height, self.image_width)).transpose((1, 2, 0))

    def step(self, action):
        if len(action.shape) == 2:
            action = np.reshape(action, (-1, 1))

        assert action.shape == self.action_dim

        action = action.astype(np.float32)
        self.env.step(action, self.state_obs, self.image_obs)

        return self.state_obs, self._reshape_image_obs()

    def set_state(self, state):
        if len(state.shape) == 2:
            state = np.reshape(state, (-1, 1))
        state = state.astype(np.float32)

        self.env.setReducedState(state)
        self.env.getObs(self.state_obs, self.image_obs)

        return self.state_obs, self._reshape_image_obs()

    def connect_unity(self):
        self.env.connectUnity()

    def disconnect_unity(self):
        self.env.disconnectUnity()
