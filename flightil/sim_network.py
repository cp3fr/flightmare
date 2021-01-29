import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import torch
from torchvision import transforms

from tqdm import tqdm
from time import time
from mpc.mpc.mpc_solver import MPCSolver
from mpc.simulation.planner import TrajectoryPlanner
from mpc.simulation.mpc_test_env import MPCTestEnv
from mpc.simulation.mpc_test_wrapper import MPCTestWrapper
from gazesim.training.utils import load_model, to_batch

from run_tests import ensure_quaternion_consistency, visualise_actions, visualise_states


class SimpleNetworkController:

    def __init__(self, model_path):
        # load model
        self.model, self.config = load_model(model_path, gpu=0, return_config=True)

        # prepare transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.config["resize"]),
            transforms.ToTensor(),
            transforms.Normalize(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
        ])

        # TODO: get info about output normalisation and stuff

    def get_action(self, img):
        # most likely have to convert to RGB since PIMS is used for loading the training
        # data, but OpenCV is used for obtaining the images from the simulation
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # apply transforms
        img = self.transform(img)

        # prepare input dictionary
        img = {"input_image_0": img}

        # run network
        out = self.model(to_batch([img]))

        # "post-process" and return output
        act = out["output_control"].cpu().detach().numpy().squeeze()
        if self.config["control_normalisation"]:
            act *= np.array([20.0, 6.0, 6.0, 6.0])

        return act


def get_mpc_trajectory(track_name, subject, run, lap):
    df_frame_index = pd.read_csv(os.path.join(os.getenv("GAZESIM_ROOT"), "index", "frame_index.csv"))
    df_mpc_gt = pd.read_csv(os.path.join(os.getenv("GAZESIM_ROOT"), "index", "drone_control_mpc_20_gt.csv"))
    df_traj_gt = pd.read_csv(os.path.join(os.getenv("GAZESIM_ROOT"), "s{:03d}".format(subject),
                                          "{:02d}_{}".format(run, track_name), "trajectory_mpc_20.csv"))

    if lap >= 0:
        match_index = (df_frame_index["track_name"] == track_name) & (df_frame_index["subject"] == subject) & \
                      (df_frame_index["run"] == run) & (df_frame_index["lap_index"] == lap)
    else:
        match_index = (df_frame_index["track_name"] == track_name) & (df_frame_index["subject"] == subject) & \
                      (df_frame_index["run"] == run)

    act = df_mpc_gt.loc[match_index, ["throttle", "roll", "pitch", "yaw"]].values
    sts = df_traj_gt[["position_x [m]", "position_y [m]", "position_z [m]",
                      "rotation_w [quaternion]", "rotation_x [quaternion]",
                      "rotation_y [quaternion]", "rotation_z [quaternion]",
                      "velocity_x [m]", "velocity_y [m]", "velocity_z [m]"]].values
    max_ts = df_traj_gt["time-since-start [s]"].iloc[-1]
    return act, sts, max_ts


if __name__ == "__main__":
    # timings
    base_frequency = 60.0 # 50.0
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
    # model_load_path = "/home/simon/gazesim-data/fpv_saliency_maps/logs/2021-01-27_13-41-06_resnet-larger-fm-mpc-10-20/checkpoints/epoch000.pt"
    # model_load_path = "/home/simon/gazesim-data/fpv_saliency_maps/logs/2021-01-27_14-14-01_resnet-larger-fm-mpc-10-20-no-out-norm/checkpoints/epoch001.pt"
    # model_load_path = "/home/simon/gazesim-data/fpv_saliency_maps/logs/2021-01-27_20-32-06_resnet-larger-fm-mpc-10-20-da-correct-gt-name/checkpoints/epoch000.pt"
    model_load_path = "/home/simon/gazesim-data/fpv_saliency_maps/logs/2021-01-27_21-21-35_resnet-larger-fm-mpc-10-20-da-no-out-norm-correct-gt-name/checkpoints/epoch000.pt"

    # network expert
    # network = SimpleNetworkController(model_load_path)
    init_state = np.array([5.53432098729772, -6.7519413144563, 2.70721623458745, -0.54034852533245, -0.109890052789277,
                           -0.369566681013163, -0.7479091338427, -7.9896612581575, 9.14960496089365, 1.1539013846433])

    mpc_gt_index = 0
    mpc_actions, mpc_states, ts = get_mpc_trajectory("flat", 16, 5, 1)
    mpc_actions = mpc_actions[50:]
    print(mpc_actions[:40])

    # planning parameters
    plan_time_step = 0.1
    plan_time_horizon = 3.0

    # planner and MPC solver
    planner = TrajectoryPlanner(trajectory_path, plan_time_horizon, plan_time_step)
    mpc_solver = MPCSolver(plan_time_horizon, plan_time_step, mpc_binary_path)
    print(planner.get_initial_state())
    # exit()

    # simulation parameters (after planner to get max time)
    simulation_time_step = base_time_step
    simulation_time_horizon = total_time = planner.get_final_time_stamp()
    # total_time = ts
    switch_to_network_time = 0.0

    # environment
    env = MPCTestEnv(mpc_solver, planner, simulation_time_horizon, simulation_time_step)
    env.reset()
    # env.quad_rotor._state = mpc_states[0, :]
    # env.quad_state = mpc_states[0, :]
    # env.quad_rotor._state = init_state
    # env.quad_state = init_state

    # wrapper (always used for now?)
    wrapper = MPCTestWrapper(wave_track=False)
    # wrapper.env.setWaveTrack(False)
    wrapper.connect_unity(pub_port=10253, sub_port=10254)

    # video writer (also always used for now?)
    writer = cv2.VideoWriter(
        # "/home/simon/Desktop/flightmare_cam_test/network_resnet-larger-fm-mpc-10-20-da-no-out-norm-correct-gt-name_ep0.mp4",
        "/home/simon/Desktop/flightmare_cam_test/test_mpc_gt.mp4",
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

    # data to record
    states = []
    actions = []
    network_actions = []

    state, action, network_action, image = None, None, None, None

    # loop with "halting" (maybe change to for-loop?)
    while base_time <= total_time:
        """
        if command_time <= base_time:
            command_time += command_time_step
            # if base_time < switch_to_network_time:
            state, action = env.step()
            else:
                action = network.get_action(image)
                state, _ = env.step(action)
            # state, action = env.step()
            # if image is not None:
            #     network_action = network.get_action(image)
            # else:
            #     network_action = np.array([0.0, 0.0, 0.0, 0.0])
            # if base_time > switch_to_network_time:
            #     network.get_action(image)
            # TODO: also "calculate" and save the MPC action to see whether there is a deviation
        else:
            state, _ = env.step(action)
        """
        if base_time < switch_to_network_time:
            state, action = env.step()
        else:
            action = mpc_actions[mpc_gt_index, :]
            # print(action)
            state, _ = env.step(action)
        network_action = mpc_actions[mpc_gt_index, :]
        mpc_gt_index += 1

        if image_time <= base_time:
            image_time += image_time_step
            image = wrapper.step(state)
            writer.write(image)

        base_time += base_time_step

        # record these with the base frequency to see actions work as intended
        states.append(state)
        actions.append(action)
        network_actions.append(network_action)

        if mpc_gt_index > mpc_actions.shape[0]:
            break

    # "clean up"
    wrapper.disconnect_unity()
    writer.release()

    show_plots = True
    if show_plots:
        states = np.vstack(states)
        actions = np.vstack(actions)
        network_actions = np.vstack(network_actions)

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
        visualise_actions(actions, simulation_time_horizon, base_time_step, True, False,
                          comparison_actions=network_actions, comparison_label="(replay)")
        # plt.savefig("/home/simon/Desktop/weekly_meeting/meeting14/cam_angle_test_mpc_wave_fast_rot_actions.png")
        # plt.close()
        # visualise_actions(network_actions, simulation_time_horizon, base_time_step, True, False)

