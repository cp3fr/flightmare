import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import cv2

from pprint import pprint
from mpc.mpc.mpc_solver import MPCSolver
from mpc.simulation.planner import TrajectoryPlanner, HoverPlanner, RisePlanner
from mpc.simulation.mpc_test_env import MPCTestEnv
from mpc.simulation.mpc_test_wrapper import MPCTestWrapper

plt.style.use("ggplot")


def rotation_quaternion(x, y, z, angle_deg):
    angle_rad = np.pi * angle_deg / 180.0
    factor = np.sin(0.5 * angle_rad)

    x *= factor
    y *= factor
    z *= factor

    w = np.cos(0.5 * angle_rad)

    quat = np.array([w, x, y, z])
    # return quat / quat.sum()
    return quat


def quaternion_multiplication(quaternion_left, quaternion_right):
    r_left = quaternion_left[0]
    r_right = quaternion_right[0]
    i_left = quaternion_left[1:]
    i_right = quaternion_right[1:]

    real_result = (r_left * r_right) - np.dot(i_left, i_right)
    imaginary_result = (r_left * i_right) + (r_right * i_left) + np.cross(i_left, i_right)

    result = np.array([
        real_result,
        imaginary_result[0],
        imaginary_result[1],
        imaginary_result[2],
    ])

    return result


def row_to_state(row):
    quaternion = np.array([
        row["rotation_w [quaternion]"],
        row["rotation_x [quaternion]"],
        row["rotation_y [quaternion]"],
        row["rotation_z [quaternion]"],
    ])

    quaternion_rot = quaternion_multiplication(
        rotation_quaternion(0, 0, 1, 90),
        quaternion,
    )
    # print(np.sum(quaternion_rot))

    state = np.array([
        -row["position_y [m]"],
        row["position_x [m]"],
        row["position_z [m]"],
        quaternion_rot[0],
        quaternion_rot[1],
        quaternion_rot[2],
        quaternion_rot[3],
        # row["rotation_w [quaternion]"],
        # row["rotation_x [quaternion]"],
        # row["rotation_y [quaternion]"],
        # row["rotation_z [quaternion]"],
        -row["velocity_y [m/s]"],
        row["velocity_x [m/s]"],
        row["velocity_z [m/s]"],
    ], dtype=np.float32)

    state_original = np.array([
        row["position_x [m]"],
        row["position_y [m]"],
        row["position_z [m]"],
        row["rotation_w [quaternion]"],
        row["rotation_x [quaternion]"],
        row["rotation_y [quaternion]"],
        row["rotation_z [quaternion]"],
        row["velocity_x [m/s]"],
        row["velocity_y [m/s]"],
        row["velocity_z [m/s]"],
    ], dtype=np.float32)

    return state_original


def sample_from_trajectory(trajectory, time_stamp):
    # probably just take the closest ts for now, might do interpolation later
    index = trajectory.loc[trajectory["time-since-start [s]"] <= time_stamp, "time-since-start [s]"].idxmax()
    return row_to_state(trajectory.iloc[index])
    # return np.array([0, 0, 5, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    # return row_to_state(trajectory.iloc[0])


def ensure_quaternion_consistency(trajectory):
    trajectory = trajectory.reset_index(drop=True)
    trajectory["flipped"] = 0
    tolerance = 0.005

    quat_columns = ["rotation_{} [quaternion]".format(c) for c in ["w", "x", "y", "z"]]
    prev_quaternion = trajectory.loc[0, quat_columns]
    prev_signs_positive = prev_quaternion >= 0

    flipped = 0
    trajectory.loc[0, "flipped"] = flipped
    for i in range(1, len(trajectory.index)):
        current_quaternion = trajectory.loc[i, quat_columns]
        current_signs_positive = current_quaternion >= 0
        condition_sign = prev_signs_positive == ~current_signs_positive

        if np.sum(condition_sign) >= 3:
            # flipped = not flipped
            flipped = 1 - flipped
        trajectory.loc[i, "flipped"] = flipped

        prev_signs_positive = current_signs_positive

    # trajectory.loc[np.array(flipped) == True, quat_columns] *= -1
    trajectory.loc[trajectory["flipped"] == 1, quat_columns] *= -1.0

    """
    for i in range(1, len(trajectory.index)):
        current_quaternion = trajectory.loc[i, quat_columns]
        current_signs_positive = current_quaternion >= 0
        condition_sign = prev_signs_positive == ~current_signs_positive
        condition_zero_crossing = (np.abs(current_quaternion) < tolerance) & (np.abs(prev_quaternion) < tolerance)

        print()
        print(current_quaternion)
        print(current_signs_positive)
        print(prev_signs_positive)
        print(condition_sign)
        exit()

        if (condition_sign[0] and condition_sign[1] and condition_sign[3]) and not condition_sign[2] and np.abs(current_quaternion[2]) < tolerance:
            print(trajectory.loc[i, ["time-since-start [s]", "rotation_y [quaternion]"]])

        # if all(condition_sign | condition_zero_crossing):
        if np.sum(condition_sign) >= 3:
            # flip the quaternion
            trajectory.loc[i, quat_columns] *= -1.0
            # print(self._trajectory.loc[i, "time-since-start [s]"])
        else:
            prev_signs_positive = current_signs_positive
        prev_quaternion = trajectory.loc[i, quat_columns]
    """

    return trajectory


def visualise_states(states, trajectory, simulation_time_horizon, simulation_time_step):
    subplot_labels = ["Position [m]", "Rotation [quaternion]", "Velocity [m/s]"]
    labels = [r"$x_{pos}$", r"$y_{pos}$", r"$z_{pos}$",
              r"$q_{w}$", r"$q_{x}$", r"$q_{y}$", r"$q_{z}$",
              r"$x_{vel}$", r"$y_{vel}$", r"$z_{vel}$"]
    time_steps = np.arange(0.0, simulation_time_horizon + simulation_time_step, step=simulation_time_step)

    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(16, 8), dpi=100)
    for i in range(len(labels)):
        if i < 3:
            a = ax[0]
        elif i < 7:
            a = ax[1]
        else:
            a = ax[2]

        line = a.plot(time_steps, states[:, i], label=labels[i])
        a.plot(trajectory[:, -1], trajectory[:, i], label="{} GT".format(labels[i]),
               color=line[0].get_color(), linestyle="--")
    for a, lab in zip(ax, subplot_labels):
        a.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        a.set_ylabel(lab)
    ax[2].set_xlabel("Time [s]")
    fig.tight_layout()
    plt.show()


def test_manual_trajectory():
    # load trajectory
    trajectory = pd.read_csv("/home/simon/Downloads/trajectory_s016_r05_li01.csv")

    # create environment and set up timers
    env = MPCTestWrapper()

    time_total = trajectory["time-since-start [s]"].max()
    time_start = time.time()
    time_current = time_start

    # create video writer for the onboard camera
    writer = cv2.VideoWriter(
        "/home/simon/Desktop/flightmare_cam_test/overview_cam_test.mp4",
        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        30.0,
        (env.image_width, env.image_height)
    )

    # loop through the trajectory
    env.connect_unity()

    while (time_current - time_start) < time_total:
        time_relative = time_current - time_start

        sample = sample_from_trajectory(trajectory, time_relative)
        image = env.step(sample)

        writer.write(image)

        time_current = time.time()

    env.disconnect_unity()

    writer.release()


def test_mpc():
    # load trajectory
    trajectory_path_fast = "/home/simon/Downloads/drone.csv"
    trajectory_path_medium = "/home/simon/Downloads/trajectory_s016_r05_li01.csv"
    trajectory_path_slow = "/home/simon/Downloads/trajectory_s007_r08_li02.csv"
    trajectory = pd.read_csv(trajectory_path_fast)

    # define stuff as it would later be used
    plan_time_horizon = 2.0
    plan_time_step = 0.1
    num_plan_steps = int(plan_time_horizon / plan_time_step)

    # get some states
    planned_traj = []
    for step in range(num_plan_steps + 2):
        state = sample_from_trajectory(trajectory, step * plan_time_step).tolist()
        planned_traj += state
    planned_traj = np.array(planned_traj)

    print(planned_traj.shape)
    pprint(planned_traj.reshape((-1, 10))[:, :3])

    # construct the MPC solver and solve with these states
    mpc_solver = MPCSolver(plan_time_horizon, plan_time_step, os.path.join(os.path.abspath("../"), "mpc/mpc/saved/mpc_v2.so"))
    optimal_action, predicted_traj = mpc_solver.solve(planned_traj)

    print(optimal_action.shape, optimal_action.squeeze())
    print(predicted_traj.shape)

    pprint(predicted_traj[:, :3])


def test_planner():
    planner = TrajectoryPlanner("/home/simon/Downloads/trajectory_s016_r05_li01.csv", plan_time_horizon=0.1)
    test = planner.plan(np.zeros((10,)), 0)
    print(test.shape)
    print(test)


def test_simulation():
    # files to load
    trajectory_path_fast = "/home/simon/Downloads/drone.csv"
    trajectory_path_medium = "/home/simon/Downloads/trajectory_s016_r05_li01.csv"
    trajectory_path_slow = "/home/simon/Downloads/trajectory_s007_r08_li02.csv"
    trajectory_path = trajectory_path_slow
    mpc_binary_path = os.path.join(os.path.abspath("../"), "mpc/mpc/saved/mpc_v2.so")

    # planning parameters
    plan_time_horizon = 2.0
    plan_time_step = 0.1

    # display parameters
    use_unity = False
    show_plots = True

    # planner and MPC solver
    planner = TrajectoryPlanner(trajectory_path, plan_time_horizon, plan_time_step)
    # planner = HoverPlanner(plan_time_horizon, plan_time_step)
    # planner = RisePlanner(plan_time_horizon, plan_time_step)
    mpc_solver = MPCSolver(plan_time_horizon, plan_time_step, mpc_binary_path)

    # simulation parameters (after planner to get max time)
    simulation_time_horizon = planner.get_final_time_stamp()
    simulation_time_step = 0.02

    # print("time for trajectory:", planner.get_final_time_stamp())
    # exit()

    # environment
    env = MPCTestEnv(mpc_solver, planner, simulation_time_horizon, simulation_time_step)
    env.reset()

    wrapper = None
    if use_unity:
        wrapper = MPCTestWrapper()
        wrapper.connect_unity()

    # simulation loop
    time_prev = time.time()
    time_elapsed, steps_elapsed = 0, 0
    states = []
    while time_elapsed < env.simulation_time_horizon:
        time_elapsed = env.simulation_time_step * steps_elapsed

        # state = env.step()
        state = np.array([1.0] * 10)
        states.append(state)
        if use_unity:
            wrapper.step(state)
        # print(state[:3])

        time_current = time.time()
        # print(time_current - time_prev)
        time_prev = time.time()

        steps_elapsed += 1

    if use_unity:
        wrapper.disconnect_unity()

    if show_plots:
        states = np.vstack(states)

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

        visualise_states(states, trajectory, simulation_time_horizon, simulation_time_step)


if __name__ == "__main__":
    # test_manual_trajectory()
    # test_mpc()
    # test_planner()
    test_simulation()
