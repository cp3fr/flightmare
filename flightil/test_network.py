import os
import cv2
import numpy as np
import pandas as pd

from mpc.mpc.mpc_solver import MPCSolver
from mpc.simulation.planner import TrajectoryPlanner
from mpc.simulation.mpc_test_wrapper import RacingEnvWrapper
from sim_network import SimpleNetworkController, DDANetworkController

from run_tests import ensure_quaternion_consistency, visualise_actions, visualise_states


def test():
    # timings
    base_frequency = 60.0
    image_frequency = 60.0
    command_frequency = 60.0

    base_time_step = 1.0 / base_frequency
    image_time_step = 1.0 / image_frequency
    command_time_step = 1.0 / command_frequency

    # paths
    trajectories = [
        "/home/simon/Downloads/trajectory_s016_r05_flat_li01.csv",  # medium (median)
        "/home/simon/Downloads/trajectory_mpc_20_s016_r05_flat_li01.csv",  # medium (median) MPC
        "/home/simon/Downloads/trajectory_s024_r08_flat_li09.csv",  # fast
        "/home/simon/Downloads/trajectory_s018_r09_wave_li04.csv",  # medium wave
        "/home/simon/Downloads/trajectory_s020_r13_wave_li04.csv",  # fast wave
    ]
    trajectory_path = trajectories[0]
    model_load_path = "/home/simon/gazesim-data/fpv_saliency_maps/logs/2021-02-06_23-06-46_dda-ft-fm-mpc-60-ctrl-norm/checkpoints/epoch019.pt"
    # model_load_path = "/home/simon/gazesim-data/fpv_saliency_maps/logs/2021-02-06_23-08-39_dda-ft-fm-mpc-60-no-ctrl-norm/checkpoints/epoch019.pt"
    # model_load_path = "/home/simon/gazesim-data/fpv_saliency_maps/logs/2021-02-06_21-50-47_ue4sim-fm-mpc-60-da/checkpoints/epoch004.pt"

    # network
    dda_network = "dda" in model_load_path
    if dda_network:
        network = DDANetworkController(model_load_path)
    else:
        network = SimpleNetworkController(model_load_path)

    # planning parameters
    plan_time_step = 0.1
    plan_time_horizon = 3.0
    plan_time_step = 0.2
    plan_time_horizon = 4.0

    # planner and MPC solver
    planner = TrajectoryPlanner(trajectory_path, plan_time_horizon, plan_time_step)
    mpc_solver = MPCSolver(plan_time_horizon, plan_time_step)

    # simulation parameters (after planner to get max time)
    simulation_time_step = base_time_step
    simulation_time_horizon = total_time = planner.get_final_time_stamp()

    # Flightmare simulation/wrapper
    env = RacingEnvWrapper(wave_track=("wave" in trajectory_path))
    env.set_sim_time_step(simulation_time_step)
    env.connect_unity(pub_port=10253, sub_port=10254)

    # TODO: add loop for testing multiple start times here
    switch_time = 1.0
    # for switch_time in np.arange(0.5, 5.0, step=0.5):
    #     for repetition in range(3):
    if dda_network:
        network.feature_tracker.reset()

    # video writer (also always used for now?)
    writer = cv2.VideoWriter(
        "/home/simon/Desktop/flightmare_cam_test/test_network_eval.mp4",
        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        1.0 / image_time_step,
        (env.image_width, env.image_height),
        True
    )

    # "reset" counters
    base_time = 0.0
    image_time = 0.0
    command_time = 0.0

    # data to record
    states = []
    reduced_states = []
    mpc_actions = []
    network_actions = []
    network_used = []
    time_stamps = []

    reduced_state = planner.get_initial_state()
    state = np.array(reduced_state.tolist() + ([0.0] * 15))
    image = np.zeros((env.image_height, env.image_width, 3), dtype=np.uint8)
    mpc_action, network_action, action = None, None, None
    network_used_currently = 0

    # reset environment
    env.set_reduced_state(planner.get_initial_state())

    #
    use_network = False

    # loop
    while base_time <= simulation_time_horizon:
        time_stamps.append(base_time)
        states.append(state)
        reduced_states.append(reduced_state)

        if dda_network:
            reference = planner.sample_from_trajectory(base_time, ["vel", "rot", "omega"])  # hopefully this works
            state_estimate = np.array(state[7:10].tolist() + state[3:7].tolist() + state[10:13].tolist())
            # TODO: also need to change quaternion to rotation matrix.... see in datasets how to do it
            network.append_reference(reference)
            network.append_state_estimate(state_estimate)

        # get an image BEFORE (!) moving
        if image_time <= base_time:
            image_time += image_time_step
            image = env.get_image()
            if dda_network:
                network.append_image(image, base_time)
            writer.write(image)

        # "generate" the command
        if command_time <= base_time:
            command_time += command_time_step

            planned_trajectory = np.array(planner.plan(reduced_state, base_time))
            mpc_action, predicted_trajectory, cost = mpc_solver.solve(planned_trajectory)
            if dda_network and len(network.feature_track_queue) < 8:
                network_action = np.array([np.nan, np.nan, np.nan, np.nan])
            else:
                if dda_network:
                    network_action = network.get_action()
                else:
                    network_action = network.get_action(image, state)

            action = mpc_action
            if use_network and base_time >= switch_time:
                action = network_action

            """
            if use_network:
                # decide whether to use MPC expert or network
                if base_time < switch_time:
                    planned_trajectory = np.array(planner.plan(reduced_state, base_time))
                    mpc_action, predicted_trajectory, cost = mpc_solver.solve(planned_trajectory)
                    network_action = np.array([np.nan, np.nan, np.nan, np.nan])
                    network_used_currently = 0
                else:
                    if dda_network:
                        mpc_action = network.get_action()
                    else:
                        mpc_action = network.get_action(image, state)
                    network_action = mpc_action
                    network_used_currently = 1
            else:
                planned_trajectory = np.array(planner.plan(reduced_state, base_time))
                mpc_action, predicted_trajectory, cost = mpc_solver.solve(planned_trajectory)
                if dda_network and len(network.feature_track_queue) < 8:
                    network_action = np.array([np.nan, np.nan, np.nan, np.nan])
                else:
                    if dda_network:
                        network_action = network.get_action()
                    else:
                        network_action = network.get_action(image, state)
            """

        # always take a step with the current mpc_action
        env.step(action)
        state = env.get_state()
        reduced_state = state[:10]

        mpc_actions.append(mpc_action)
        network_actions.append(network_action)
        network_used.append(network_used_currently)

        # increase time
        base_time += base_time_step
        # print(base_time)

    # states.append(state)
    # reduced_states.append(reduced_state)

    writer.release()

    states = np.vstack(states)
    reduced_states = np.vstack(reduced_states)
    mpc_actions = np.vstack(mpc_actions)
    network_actions = np.vstack(network_actions)

    show_plots = True
    if show_plots:
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

        visualise_states(reduced_states, trajectory, simulation_time_horizon, base_time_step, True, False)
        if use_network:
            visualise_actions(mpc_actions, simulation_time_horizon, base_time_step, True, False)
        else:
            visualise_actions(mpc_actions, simulation_time_horizon, base_time_step, True, False,
                              comparison_actions=network_actions, comparison_label="(network)")

    save_data = False
    if save_data:
        data = {
            "time-since-start [s]": time_stamps,
            "throttle_mpc": mpc_actions[:, 0],
            "roll_mpc": mpc_actions[:, 1],
            "pitch_mpc": mpc_actions[:, 2],
            "yaw_mpc": mpc_actions[:, 3],
            "throttle_nw": network_actions[:, 0],
            "roll_nw": network_actions[:, 1],
            "pitch_nw": network_actions[:, 2],
            "yaw_nw": network_actions[:, 3],
            "position_x [m]": states[:, 0],
            "position_y [m]": states[:, 1],
            "position_z [m]": states[:, 2],
            "rotation_w [quaternion]": states[:, 3],
            "rotation_x [quaternion]": states[:, 4],
            "rotation_y [quaternion]": states[:, 5],
            "rotation_z [quaternion]": states[:, 6],
            "velocity_x [m]": states[:, 7],
            "velocity_y [m]": states[:, 8],
            "velocity_z [m]": states[:, 9],
            "omega_x [rad/s]": states[:, 10],
            "omega_y [rad/s]": states[:, 11],
            "omega_z [rad/s]": states[:, 12],
            "network_used": network_used
        }
        data = pd.DataFrame(data)
        data.to_csv(os.path.join("/home/simon/Desktop/weekly_meeting/meeting17",
                                 "trajectory_mpc2nw_st-{:02d}_if-{:02d}_cf-{:02d}_{:02d}.csv"
                                 .format(int(switch_time * 10), int(image_frequency),
                                         int(command_frequency), repetition)), index=False)


if __name__ == "__main__":
    test()
    # TODO: add CLI argument stuff instead of hard-coding
