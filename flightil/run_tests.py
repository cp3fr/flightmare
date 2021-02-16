import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import cv2

from tqdm import tqdm
from pprint import pprint
from mpc.mpc.mpc_solver import MPCSolver
from mpc.simulation.planner import TrajectoryPlanner, HoverPlanner, RisePlanner
from mpc.simulation.mpc_test_env import MPCTestEnv
from mpc.simulation.mpc_test_wrapper import MPCTestWrapper, RacingEnvWrapper
# from envs.racing_env_wrapper import RacingEnvWrapper
from features.feature_tracker import FeatureTracker

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
        row["position_z [m]"],  # + 0.75,
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
        row["position_z [m]"],  # + 0.75,
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
    if time_stamp < trajectory["time-since-start [s]"].min():
        index = 0
    else:
        index = trajectory.loc[trajectory["time-since-start [s]"] <= time_stamp, "time-since-start [s]"].idxmax()
    return row_to_state(trajectory.iloc[index])
    # return np.array([0, 0, 5, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    # return row_to_state(trajectory.iloc[0])


def ensure_quaternion_consistency(trajectory, use_norm=True):
    trajectory = trajectory.reset_index(drop=True)
    flipped = 0
    trajectory["flipped"] = 0
    trajectory.loc[0, "flipped"] = flipped

    quat_columns = ["rotation_{} [quaternion]".format(c) for c in ["w", "x", "y", "z"]]
    prev_quaternion = trajectory.loc[0, quat_columns]
    prev_signs_positive = prev_quaternion >= 0

    norm_diffs = []
    for i in range(1, len(trajectory.index)):
        current_quaternion = trajectory.loc[i, quat_columns]
        current_signs_positive = current_quaternion >= 0
        condition_sign = prev_signs_positive == ~current_signs_positive

        # TODO: should probably do something like X standard deviations above the running mean for "good methodology"
        norm_diff = np.linalg.norm(prev_quaternion.values - current_quaternion.values)
        norm_diffs.append(norm_diff)

        if use_norm:
            if norm_diff >= 0.5:  # TODO should this be 1.0?
                flipped = 1 - flipped
        else:
            if np.sum(condition_sign) >= 3:
                flipped = 1 - flipped
        trajectory.loc[i, "flipped"] = flipped

        prev_signs_positive = current_signs_positive
        prev_quaternion = current_quaternion

    trajectory.loc[trajectory["flipped"] == 1, quat_columns] *= -1.0

    return trajectory, norm_diffs


def visualise_states(states, trajectory, simulation_time_horizon, simulation_time_step, exclude_first=False, skip_show=False):
    subplot_labels = ["Position [m]", "Rotation [quaternion]", "Velocity [m/s]"]
    labels = [r"$x_{pos}$", r"$y_{pos}$", r"$z_{pos}$",
              r"$q_{w}$", r"$q_{x}$", r"$q_{y}$", r"$q_{z}$",
              r"$x_{vel}$", r"$y_{vel}$", r"$z_{vel}$"]
    time_start = simulation_time_step if exclude_first else 0.0
    time_steps = np.arange(time_start, simulation_time_horizon + simulation_time_step, step=simulation_time_step)
    time_steps = time_steps[:len(states)]

    print(time_start)
    print(simulation_time_horizon)
    print(simulation_time_step)
    print(time_steps.shape)

    # states = states[:100]
    # trajectory = trajectory[:100]
    # time_steps = time_steps[:100]

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
    if not skip_show:
        plt.show()


def visualise_actions(actions, simulation_time_horizon, simulation_time_step, exclude_first=False, skip_show=False,
                      comparison_actions=None, comparison_label="comp"):
    labels = ["thrust", "roll", "pitch", "yaw"]
    time_start = simulation_time_step if exclude_first else 0.0
    time_steps = np.arange(time_start, simulation_time_horizon + simulation_time_step, step=simulation_time_step)
    time_steps = time_steps[:len(actions)]

    # actions = actions[:100]
    # time_steps = time_steps[:100]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 3), dpi=100)
    for i in range(len(labels)):
        line = ax.plot(time_steps, actions[:, i], label=labels[i])
        if comparison_actions is not None:
            ax.plot(time_steps, comparison_actions[:, i], label="{} {}".format(labels[i], comparison_label),
                    color=line[0].get_color(), linestyle="--")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.set_ylabel("Control input")
    ax.set_xlabel("Time [s]")
    fig.tight_layout()
    if not skip_show:
        plt.show()


def test_manual_trajectory():
    # load trajectory
    trajectory_path_flat_fast = "/home/simon/Downloads/drone.csv"
    trajectory_path_flat_fast_human = "/home/simon/Downloads/trajectory_s024_r08_flat_li09.csv"
    trajectory_path_flat_medium = "/home/simon/Downloads/trajectory_s016_r05_flat_li01.csv"
    trajectory_path_flat_slow = "/home/simon/Downloads/trajectory_s007_r08_flat_li02.csv"

    trajectory_path_wave_medium = "/home/simon/Downloads/trajectory_s018_r09_wave_li04.csv"
    trajectory_path_wave_slow = "/home/simon/Downloads/trajectory_s010_r13_wave_li03.csv"
    trajectory_path_wave_slowest = "/home/simon/Downloads/trajectory_s007_r14_wave_li00.csv"

    trajectory_path = trajectory_path_flat_medium
    trajectory = pd.read_csv(trajectory_path)

    # set the mode
    render_real_time = False
    skip = 5
    if render_real_time:
        fps = 25.0
    else:
        frame_times = (np.array(trajectory["time-since-start [s]"]) -
                       np.roll(trajectory["time-since-start [s]"], shift=1))[1:]
        fps = 60.0

    # create environment and set up timers
    env = MPCTestWrapper(wave_track=False)

    time_total = trajectory["time-since-start [s]"].max()

    # create video writer for the onboard camera
    writer = cv2.VideoWriter(
        # "/home/simon/Desktop/flightmare_cam_test/alphapilot_arena_test.mp4",
        # "/home/simon/Desktop/flightmare_cam_test/fov_test/final_colours_9.mp4",
        # "/home/simon/Desktop/flightmare_cam_test/flightmare_original.mp4",
        "/home/simon/Desktop/flightmare_cam_test/flightmare_original_fov_74.mp4",
        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        fps,
        (env.image_width, env.image_height),
        True
    )

    # loop through the trajectory
    env.connect_unity()

    if render_real_time:
        time_start = time.time()
        time_current = time_start
        while (time_current - time_start) < time_total:
            time_relative = time_current - time_start

            sample = sample_from_trajectory(trajectory, time_relative)
            image = env.step(sample)

            writer.write(image)

            time_current = time.time()
    else:
        for _ in range(20):
            sample = sample_from_trajectory(trajectory, 0.0)
            image = env.step(sample)
        time_current = 0.0
        time_step = 1.0 / fps
        while time_current <= time_total:
            sample = sample_from_trajectory(trajectory, time_current)
            image = env.step(sample)
            writer.write(image)
            time_current += time_step

    env.disconnect_unity()

    writer.release()


def test_mpc():
    # load trajectory
    trajectory_path_flat_fast = "/home/simon/Downloads/drone.csv"
    trajectory_path_flat_medium = "/home/simon/Downloads/trajectory_s016_r05_flat_li01.csv"
    trajectory_path_flat_slow = "/home/simon/Downloads/trajectory_s007_r08_flat_li02.csv"
    trajectory = pd.read_csv(trajectory_path_flat_fast)

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
    mpc_solver = MPCSolver(plan_time_horizon, plan_time_step, os.path.join(os.path.abspath("/"), "mpc/mpc/saved/mpc_v2.so"))
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
    trajectory_path_flat_fast = "/home/simon/Downloads/drone.csv"
    trajectory_path_flat_fast_human = "/home/simon/Downloads/trajectory_s024_r08_flat_li09.csv"
    trajectory_path_flat_medium = "/home/simon/Downloads/trajectory_s016_r05_flat_li01.csv"
    trajectory_path_flat_slow = "/home/simon/Downloads/trajectory_s007_r08_flat_li02.csv"

    trajectory_path_wave_medium = "/home/simon/Downloads/trajectory_s018_r09_wave_li04.csv"
    trajectory_path_wave_slow = "/home/simon/Downloads/trajectory_s010_r13_wave_li03.csv"
    trajectory_path_wave_slowest = "/home/simon/Downloads/trajectory_s007_r14_wave_li00.csv"

    trajectory_path = trajectory_path_flat_medium
    mpc_binary_path = os.path.join(os.path.abspath("/"), "mpc/mpc/saved/mpc_v2.so")

    # planning parameters
    plan_time_horizon = 3.0
    plan_time_step = 0.1

    # display parameters
    use_unity = True
    show_plots = False
    write_video = True

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
        wrapper = MPCTestWrapper(wave_track=False)
        wrapper.connect_unity()

    # video writer
    writer = None
    if use_unity and write_video:
        writer = cv2.VideoWriter(
            "/home/simon/Desktop/flightmare_cam_test/arena_test.mp4",
            cv2.VideoWriter_fourcc("m", "p", "4", "v"),
            1.0 / simulation_time_step,
            (wrapper.image_width, wrapper.image_height),
            True
        )

    # simulation loop
    time_prev = time.time()
    time_elapsed, steps_elapsed = 0, 0
    states = []
    actions = []
    image = None
    while time_elapsed < env.simulation_time_horizon:
        time_elapsed = env.simulation_time_step * steps_elapsed

        state, action = env.step()
        # state = np.array([1.0] * 10)
        states.append(state)
        actions.append(action)
        if use_unity:
            image = wrapper.step(state)
        # print(state[:3])

        if use_unity and write_video:
            writer.write(image)

        time_current = time.time()
        # print(time_current - time_prev)
        time_prev = time.time()

        steps_elapsed += 1

    if use_unity:
        wrapper.disconnect_unity()

    if use_unity and write_video:
        writer.release()

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

        visualise_states(states, trajectory, simulation_time_horizon, simulation_time_step)
        visualise_actions(actions, simulation_time_horizon, simulation_time_step)


def test_feature_tracker():
    video_path = "/home/simon/Desktop/weekly_meeting/meeting11/video_flat_medium_s016_r05_flat_li01.mp4"
    # video_path = "/home/simon/Desktop/weekly_meeting/meeting11/flat_medium_original.mp4"
    video_path = "/home/simon/Desktop/weekly_meeting/meeting12/flat_medium_original.mp4"
    flightmare_video_path = "/home/simon/Desktop/weekly_meeting/meeting12/flat_medium_alphapilot_arena.mp4"

    # some parameters
    show_tracks = True
    show_plots = True
    write_video = True

    # video stuff
    video_capture = cv2.VideoCapture(video_path)
    flightmare_video_capture = cv2.VideoCapture(flightmare_video_path)
    w, h, fps, fourcc, num_frames = (video_capture.get(i) for i in range(3, 8))
    video_writer = None
    if write_video:
        video_writer = cv2.VideoWriter(
            "/home/simon/Desktop/flightmare_cam_test/features_alphapilot_arena_test.mp4",
            # "/home/simon/Desktop/weekly_meeting/meeting11/video_flat_medium_features.mp4",
            int(fourcc),
            fps,
            (int(w), int(h)),
            True
        )

    # mask to cut out the numbers in the lower right
    _, frame = video_capture.read()
    corner_mask = np.full_like(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 255)
    corner_mask[530:, 700:] = 0
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    flightmare_video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # create the trackers
    features_to_track = 100
    tracker = FeatureTracker(max_features_to_track=features_to_track, static_mask=corner_mask)
    flightmare_tracker = FeatureTracker(max_features_to_track=features_to_track, static_mask=corner_mask)

    # create some random colors
    np.random.seed(127)
    colors = np.random.randint(0, 255, (tracker.max_features_to_track, 3))

    time_steps = []
    tracking_min = []
    tracking_max = []
    tracking_medians = []
    tracking_means = []
    tracking_stds = []
    counts = []

    flightmare_tracking_min = []
    flightmare_tracking_max = []
    flightmare_tracking_medians = []
    flightmare_tracking_means = []
    flightmare_tracking_stds = []
    flightmare_counts = []

    counter = 0
    # TODO: how can we actually draw "persisting" feature tracks?
    #  => probably need dictionary? but when should stuff be removed?
    test_dict = {}
    while True:
        ret, frame = video_capture.read()
        flightmare_ret, flightmare_frame = flightmare_video_capture.read()
        # if not ret:
        if not (ret and flightmare_ret):
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flightmare_frame_gray = cv2.cvtColor(flightmare_frame, cv2.COLOR_BGR2GRAY)

        # extract the features
        features, previous_points, current_points, matched_points = tracker.process_image(frame_gray, return_image_points=True)
        flightmare_features = flightmare_tracker.process_image(flightmare_frame_gray)

        print(features.shape)

        if show_tracks:
            for f_idx, f in enumerate(features):
                if int(f[0]) not in test_dict:
                    test_dict[int(f[0])] = [current_points[f_idx, :]]
                else:
                    test_dict[int(f[0])].append(current_points[f_idx, :])
        else:
            previous_points = previous_points[:matched_points]
            current_points = current_points[:matched_points]
        """
        previous_points, current_points = np.array([]), np.array([])
        for t_idx, t in enumerate(trackers):
            if t_idx == 0:
                features, previous_points, current_points = t.process_image(frame_gray, return_image_points=True)
            else:
                features = t.process_image(frame_gray)

            tracking_means[t_idx].append(np.mean(features[:, 1]))
            tracking_stds[t_idx].append(np.std(features[:, 1]))
            counts[t_idx].append(features.shape[0])
        """

        time_steps.append(counter * (1.0 / fps))
        tracking_min.append(np.min(features[:, 1]))
        tracking_max.append(np.max(features[:, 1]))
        tracking_medians.append(np.median(features[:, 1]))
        tracking_means.append(np.mean(features[:, 1]))
        tracking_stds.append(np.std(features[:, 1]))
        counts.append(features.shape[0])

        flightmare_tracking_min.append(np.min(flightmare_features[:, 1]))
        flightmare_tracking_max.append(np.max(flightmare_features[:, 1]))
        flightmare_tracking_medians.append(np.median(flightmare_features[:, 1]))
        flightmare_tracking_means.append(np.mean(flightmare_features[:, 1]))
        flightmare_tracking_stds.append(np.std(flightmare_features[:, 1]))
        flightmare_counts.append(flightmare_features.shape[0])

        mask = np.zeros_like(frame)
        if show_tracks:
            for f_idx, f in enumerate(features):
                points = test_dict[int(f[0])]
                for i in range(len(points) - 1):
                    mask = cv2.line(mask, (points[i][0], points[i][1]), (points[i + 1][0], points[i + 1][1]),
                                    colors[f_idx].tolist(), 2)
                frame = cv2.circle(frame, (points[-1][0], points[-1][1]), 5, colors[f_idx].tolist(), -1)
        else:
            # draw the feature points
            for i, (new, old) in enumerate(zip(current_points, previous_points)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (a, b), (c, d), colors[i].tolist(), 2)
                frame = cv2.circle(frame, (a, b), 5, colors[i].tolist(), -1)
        frame = cv2.add(frame, mask)

        if write_video:
            video_writer.write(frame)

        cv2.imshow("frame", frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        counter += 1

    video_capture.release()
    flightmare_video_capture.release()
    if write_video:
        video_writer.release()

    if show_plots:
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(16, 8), dpi=100)

        line = ax[0].plot(time_steps, tracking_medians, label="Median")
        ax[0].plot(time_steps, flightmare_tracking_medians, label="Median (Flightmare)",
                   color=line[0].get_color(), linestyle="--")
        line = ax[0].plot(time_steps, tracking_max, label="Maximum")
        ax[0].plot(time_steps, flightmare_tracking_max, label="Maximum (Flightmare)",
                   color=line[0].get_color(), linestyle="--")
        line = ax[0].plot(time_steps, tracking_min, label="Minimum")
        ax[0].plot(time_steps, flightmare_tracking_min, label="Minimum (Flightmare)",
                   color=line[0].get_color(), linestyle="--")
        ax[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        ax[0].set_ylabel("#frames tracked")

        ax[1].plot(time_steps, counts, label="Original")
        ax[1].plot(time_steps, flightmare_counts, label="Flightmare", linestyle="--")
        ax[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        ax[1].set_ylabel("#features")
        ax[1].set_xlabel("Time [s]")
        ax[1].set_ylim(bottom=0)

        fig.tight_layout()
        plt.show()
        """
        for t_idx in range(len(trackers)):
            ax[0].plot(time_steps, tracking_means[t_idx], label="[{}]".format(features_to_track[t_idx]))
            ax[1].plot(time_steps, tracking_stds[t_idx], label="[{}]".format(features_to_track[t_idx]))
            ax[2].plot(time_steps, counts[t_idx], label="[{}]".format(features_to_track[t_idx]))
        for a, lab in zip(ax, ["#frames tracked mean", "#frames tracked std", "#features"]):
            a.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
            a.set_ylabel(lab)
        """


def test_features():
    video_path = "/home/simon/gazesim-data/fpv_saliency_maps/data/center_start/s016/05_flat/screen.mp4"
    # video_path = "/home/simon/Desktop/flightmare_cam_test/cam_positioning_test.mp4"
    cap = cv2.VideoCapture(video_path)

    max_features = 50

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=max_features,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(21, 21),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    np.random.seed(127)
    color = np.random.randint(0, 255, (max_features, 3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    corner_mask = np.ones_like(old_gray) * 255
    corner_mask[530:, 700:] = 0
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=corner_mask, **feature_params)
    p0 = p0.reshape(-1, 2)[:max_features]

    temp = []
    for p in p0:
        if 0 <= p[0] < old_gray.shape[1] and 0 <= p[1] < old_gray.shape[0]:
            temp.append(p)
    p0 = np.vstack(temp)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while True:
        mask = np.zeros_like(old_frame)

        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # TODO: also need to look for good new points to track
        # TODO: add IDs and only keep those lines that still have an ID
        print(len(p0))

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        st = st.reshape(st.shape[0])
        p1 = p1.reshape(-1, 2)

        for p_idx, p in enumerate(p1):
            if not (0 <= p[0] < old_gray.shape[1] and 0 <= p[1] < old_gray.shape[0]):
                st[p_idx] = 0

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)

        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # find new features if there aren't enough old ones
        features_left = max_features - good_new.shape[0]
        if features_left > 0:
            feature_mask = np.ones_like(frame_gray) * 255
            for point in good_new:
                feature_mask = cv2.circle(feature_mask, tuple(point), 7, 0, -1)
                # feature_mask[int(point[1]), int(point[0])] = 0
            feature_mask = cv2.bitwise_and(feature_mask, corner_mask)
            additional_features = cv2.goodFeaturesToTrack(frame_gray, mask=feature_mask, **feature_params)

            if additional_features is not None:
                additional_features = additional_features.reshape(-1, 2)
                good_new = np.concatenate((good_new, additional_features), axis=0)

        good_new = good_new[:max_features]

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new

    cv2.destroyAllWindows()
    cap.release()


def test_gate_size():
    # basically just place a single gate at the origin and try to estimate its size by moving a quadcopter
    # (used only as a dummy) e.g. between 0 and 1 meter in one direction

    writer = cv2.VideoWriter(
        "/home/simon/Desktop/flightmare_cam_test/arena_show_case.mp4",
        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        60.0,
        (800, 600),
        True
    )

    wrapper = MPCTestWrapper(wave_track=False)
    wrapper.connect_unity()

    positions = [
        np.array([-30.0, -16.0, 7.0, 0.88104, -0.1061931, 0.3557709, 0.2931188, 0.0, 0.0, 0.0]),
        np.array([-30.0, -16.0, 7.0, 0.88104, -0.1061931, 0.3557709, 0.2931188, 0.0, 0.0, 0.0]),
        # np.array([1.0, 0.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]

    last_switch = time.time()
    step = 3.0
    temp = 0
    c = 0
    while c < 90:
        current = time.time()
        diff = current - last_switch
        if diff > step:
            last_switch = current
            temp = 1 - temp
        image = wrapper.step(positions[temp])
        writer.write(image)
        cv2.imwrite("/home/simon/Desktop/flightmare_cam_test/arena_show_case.png", image)
        time.sleep(0.05)
        c += 1

    wrapper.disconnect_unity()
    writer.release()


def test_racing_env():
    positions = [
        np.array([-30.0, -16.0, 7.0, 0.88104, -0.1061931, 0.3557709, 0.2931188, 0.0, 0.0, 0.0]),
        np.array([-20.0, -11.0, 7.0, 0.88104, -0.1061931, 0.3557709, 0.2931188, 0.0, 0.0, 0.0]),
        np.array([-10.0, -7.0, 7.0, 0.88104, -0.1061931, 0.3557709, 0.2931188, 0.0, 0.0, 0.0]),
        np.array([0.0, -2.0, 7.0, 0.88104, -0.1061931, 0.3557709, 0.2931188, 0.0, 0.0, 0.0]),
    ]

    writer = cv2.VideoWriter(
        "/home/simon/Desktop/flightmare_cam_test/test_racing_env.mp4",
        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        20.0,
        (800, 600),
        True
    )

    wrapper = RacingEnvWrapper(wave_track=False)
    wrapper.connect_unity()

    last_switch = time.time()
    step = 3.0
    temp = 0
    c = 0
    while c < 300:
        current = time.time()
        diff = current - last_switch
        if diff > step:
            last_switch = current
            temp = (temp + 1) % len(positions)
        wrapper.set_reduced_state(positions[temp])
        image = wrapper.get_image()
        writer.write(image)
        time.sleep(0.05)
        c += 1

    wrapper.disconnect_unity()
    writer.release()


if __name__ == "__main__":
    # test_manual_trajectory()
    # test_mpc()
    # test_planner()
    test_simulation()
    # test_features()
    # test_feature_tracker()
    # test_gate_size()
    # test_racing_env()

