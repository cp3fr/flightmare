import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from tqdm import tqdm
from time import time
from mpc.mpc.mpc_solver import MPCSolver
from mpc.simulation.planner import TrajectoryPlanner
from mpc.simulation.mpc_test_env import MPCTestEnv
from mpc.simulation.mpc_test_wrapper import MPCTestWrapper

from run_tests import ensure_quaternion_consistency, visualise_actions, visualise_states


# simulation interface (should provide getter methods for sensor data and accept commands of some form)
class SimulationInterface:

    def __init__(self):
        # should take care of how the simulation is actually run (whether in Python or Flightmare)
        self.last_image = None
        self.last_state = None
        self.last_imu = None

    def step(self, command, return_observation=False):
        pass

    def get_image(self):
        return self.last_image

    def get_state(self):
        return self.last_state

    def get_imu(self):
        return self.last_imu


# command "determination" interface (should give new commands => what information does it need/get => only state?)
class CommandInterface:

    def __init__(self):
        # should e.g. contain
        # - trajectory planner (both for training and testing I guess)
        # - MPC solver and/or network?

        self.last_state = None

    def get_command(self, state):
        # state should be a dict, I guess?
        self.last_state = state
        return "Hemlo"


if __name__ == "__main__":
    # timings
    base_frequency = 60.0
    state_frequency = 50.0
    image_frequency = 50.0
    command_frequency = 25.0

    base_time_step = 1.0 / base_frequency
    state_time_step = 1.0 / state_frequency
    image_time_step = 1.0 / image_frequency
    command_time_step = 1.0 / command_frequency

    # paths
    trajectory_path = "/home/simon/Downloads/trajectory_s016_r05_flat_li01.csv"  # medium (median)
    # trajectory_path = "/home/simon/Downloads/trajectory_s024_r08_flat_li09.csv"  # fast
    # trajectory_path = "/home/simon/Downloads/trajectory_s018_r09_wave_li04.csv"  # medium wave
    # trajectory_path = "/home/simon/Downloads/trajectory_s020_r13_wave_li04.csv"  # fast wave
    mpc_binary_path = os.path.join(os.path.abspath("../"), "mpc/mpc/saved/mpc_v2.so")

    # planning parameters
    plan_time_step = 0.1
    plan_time_horizon = 3.0

    # planner and MPC solver
    planner = TrajectoryPlanner(trajectory_path, plan_time_horizon, plan_time_step)
    mpc_solver = MPCSolver(plan_time_horizon, plan_time_step, mpc_binary_path)

    # simulation parameters (after planner to get max time)
    simulation_time_step = base_time_step
    simulation_time_horizon = total_time = planner.get_final_time_stamp()

    # environment
    env = MPCTestEnv(mpc_solver, planner, simulation_time_horizon, simulation_time_step)
    env.reset()

    # wrapper (always used for now?)
    wrapper = MPCTestWrapper(wave_track=False)
    # wrapper.env.setWaveTrack(False)
    wrapper.connect_unity(pub_port=10253, sub_port=10254)

    # video writer (also always used for now?)
    writer = cv2.VideoWriter(
        # "/home/simon/Desktop/flightmare_cam_test/alphapilot_arena_mpc_async_5.mp4",
        # "/home/simon/Desktop/weekly_meeting/meeting14/cam_angle_test_mpc_wave_fast_rot.mp4",
        "/home/simon/Desktop/flightmare_cam_test/test_headless_mode.mp4",
        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        1.0 / image_time_step,
        (wrapper.image_width, wrapper.image_height),
        True
    )

    # objects (interfaces to be implemented, but not used yet)
    """
    simulation_interface = SimulationInterface()
    command_interface = CommandInterface()

    current_state = {"state": None, "image": None, "imu": None}
    current_command = None
    """

    # "reset" counters
    base_time = 0.0
    state_time = 0.0
    image_time = 0.0
    command_time = 0.0

    counter = [0, 0, 0, 0]

    # data to record
    states = []
    actions = []

    # TODO: need to separate action computation and step in env!!!
    state, action, image = None, None, None

    # loop with "halting" (maybe change to for-loop?)
    total_time_command = 0.0
    total_time_image = 0.0
    while base_time <= total_time:
        start_command = time()
        if command_time <= base_time:
            # current_command = command_interface.get_command(current_state)
            # simulation_interface.step(current_command)
            command_time += command_time_step
            counter[3] += 1

            # not using the new interfaces
            # => only now will a new action be computed
            state, action = env.step()
            # actions.append(action)
        else:
            # not using the new interfaces
            state, _ = env.step(action)
        total_time_command += time() - start_command

        if state_time <= base_time:
            # current_state["state"] = simulation_interface.get_state()
            state_time += state_time_step
            counter[1] += 1

            # not using the new interfaces
            # states.append(state)

        start_image = time()
        if image_time <= base_time:
            # current_state["image"] = simulation_interface.get_image()
            image_time += image_time_step
            counter[2] += 1

            # not using the new interfaces; TODO: need separate get_image and get_imu (see interfaces above)
            image = wrapper.step(state)
            writer.write(image)
        total_time_image += time() - start_image

        base_time += base_time_step
        counter[0] += 1

        # record these with the base frequency to see actions work as intended
        actions.append(action)
        states.append(state)

    print("results for {} FPS".format(image_frequency))
    print("command - total: {}s, average: {}s".format(total_time_command, total_time_command / counter[3]))
    print(" image  - total: {}s, average: {}s".format(total_time_image, total_time_image / counter[2]))

    # "clean up"
    wrapper.disconnect_unity()
    writer.release()

    print(counter[0], base_time)
    print(counter[1], state_time)
    print(counter[2], image_time)
    print(counter[3], command_time)

    show_plots = True
    if show_plots:
        states = np.vstack(states)
        actions = np.vstack(actions)

        trajectory = pd.read_csv(trajectory_path)
        trajectory = trajectory[trajectory["time-since-start [s]"] <= simulation_time_horizon]
        trajectory = ensure_quaternion_consistency(trajectory)
        trajectory = trajectory[[
            "position_x [m]",
            "position_y [m]",
            "position_z [m]",
            "rotation_w [quaternion]",
            "rotation_x [quaternion]",
            "rotation_y [quaternion]",
            "rotation_z [quaternion]",
            "velocity_x [m/s]",
            "velocity_y [m/s]",
            "velocity_z [m/s]",
            "time-since-start [s]",
        ]].values

        visualise_states(states, trajectory, simulation_time_horizon, base_time_step, True, False)
        # plt.savefig("/home/simon/Desktop/weekly_meeting/meeting14/cam_angle_test_mpc_wave_fast_rot_states.png")
        # plt.close()
        visualise_actions(actions, simulation_time_horizon, base_time_step, True, False)
        # plt.savefig("/home/simon/Desktop/weekly_meeting/meeting14/cam_angle_test_mpc_wave_fast_rot_actions.png")
        # plt.close()

