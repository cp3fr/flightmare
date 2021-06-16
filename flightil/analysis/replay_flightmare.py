import os
import numpy as np
import pandas as pd
import cv2
import argparse
import time
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from envs.racing_env_wrapper import RacingEnvWrapper

REDUCED_STATE_VARS = [
    "position_x [m]", "position_y [m]", "position_z [m]",
    "rotation_w [quaternion]", "rotation_x [quaternion]", "rotation_y [quaternion]", "rotation_z [quaternion]",
]


def plot_gates_3d(
        track: pd.DataFrame,
        ax: plt.axis=None,
        color: str='b',
        width: float=4,
        ) -> plt.axis:
    """
    Plot gates as rectangles in 3D.
    """
    if ax is None:
        fig = plt.figure()
        fig.set_figwidth(20)
        fig.set_figheight(20)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
    # loop over gates
    for igate in range(track.shape[0]):
        # gate center
        position = track.loc[:, ('px', 'py', 'pz')].iloc[igate].values
        # gate rotation
        rotation = track.loc[:, ('qx', 'qy', 'qz', 'qw')].iloc[igate].values
        # checkpoint center
        checkpoint_center = position
        # checkpoint size
        checkpoint_size = track.loc[:, ('dx', 'dy', 'dz')].iloc[igate].values
        # loop over axes
        corners = np.empty((0, 3))

        for y, z in [
            (-1, 1),
            (1, 1),
            (1, -1),
            (-1, -1),
            (-1, 1),
            ]:
                # determine gate corner by: 1. add half the xyz size to checkpoint center, 2. rotate according to rotation quaternion
                corner = Rotation.from_quat(rotation).apply(
                    np.array([0,
                              y * checkpoint_size[1] / 2,
                              z * checkpoint_size[2] / 2])).reshape((1, -1))

                corners = np.vstack((corners,
                                     corner))
        # plot current corner
        ax.plot(checkpoint_center[0] + corners[:, 0],
                checkpoint_center[1] + corners[:, 1],
                checkpoint_center[2] + corners[:, 2],
                color=color, linewidth=width)
    return ax


def main(args):
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # load the trajectory
    trajectory_data = pd.read_csv(args.trajectory_path)
    trajectory_data = trajectory_data.iloc[200:]
    trajectory_data = trajectory_data.iloc[::5]
    # trajectory_data["position_x [m]"] -= 0.35
    # print(trajectory_data.columns)
    # exit()

    # load the reference, assumed to be in the same folder
    reference_data = pd.read_csv(os.path.join(os.path.dirname(args.trajectory_path), "original.csv"))

    # determine the rate of recording
    time_stamps = trajectory_data["time-since-start [s]"].values
    time_stamp_diff = np.nanmedian(np.diff(time_stamps))

    skip = int((1 / time_stamp_diff) / 20)

    # determine whether we should render in Flightmare
    flightmare_wrapper = None
    flightmare_video_writer = None
    if "flightmare" in args.outputs:
        flightmare_wrapper = RacingEnvWrapper(wave_track="wave" in args.trajectory_path)
        flightmare_wrapper.connect_unity(args.pub_port, args.sub_port)

        flightmare_video_writer = cv2.VideoWriter(
            os.path.join(args.output_path, "flightmare_fpv.mp4"),
            cv2.VideoWriter_fourcc("m", "p", "4", "v"),
            float(int(1.0 / time_stamp_diff)),
            (flightmare_wrapper.image_width, flightmare_wrapper.image_height),
            True,
        )

        # wait until Unity rendering/image queue has calmed down
        for _ in range(50):
            flightmare_wrapper.get_image()
            time.sleep(0.1)

    # loop through each step
    if "flightmare" in args.outputs:
        for idx, row in tqdm(trajectory_data.iterrows(), total=len(trajectory_data.index)):
            flightmare_wrapper.set_reduced_state(row[REDUCED_STATE_VARS].values)
            flightmare_frame = flightmare_wrapper.get_image()
            flightmare_video_writer.write(flightmare_frame)

    if "anim" in args.outputs:
        trajectory_data = trajectory_data.iloc[np.arange(0, trajectory_data.shape[0], skip), :]
        reference_data = reference_data.iloc[np.arange(0, reference_data.shape[0], skip), :]

        track = pd.read_csv("./tracks/{}.csv".format("wave" if "wave" in args.trajectory_path else "flat"))
        rename_dict = {
            'pos_x': 'px',
            'pos_y': 'py',
            'pos_z': 'pz',
            'rot_x_quat': 'qx',
            'rot_y_quat': 'qy',
            'rot_z_quat': 'qz',
            'rot_w_quat': 'qw',
            'dim_x': 'dx',
            'dim_y': 'dy',
            'dim_z': 'dz',
        }
        track = track.rename(columns=rename_dict)
        track = track[list(rename_dict.values())]
        track['pz'] += 0.35
        track['dx'] = 0.
        track['dy'] = 3
        track['dz'] = 3

        fig = plt.figure(figsize=(19, 10), dpi=100)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.view_init(elev=30 - (20 if "wave" in args.trajectory_path else 0), azim=270)
        ax.set_xlim((-30, 30))
        ax.set_ylim((-20, 20))
        ax.set_zlim((-1, 5 + (3 if "wave" in args.trajectory_path else 0)))
        ax.set_box_aspect((6, 4, 1))
        if not args.anim_grid:
            plt.axis("off")

        ax.plot(reference_data["position_x [m]"].values,
                reference_data["position_y [m]"].values,
                reference_data["position_z [m]"].values, lw=2, c="#fcba03")

        line, = ax.plot(trajectory_data["position_x [m]"].values,
                        trajectory_data["position_y [m]"].values,
                        trajectory_data["position_z [m]"].values, lw=3, c="k")

        ax = plot_gates_3d(track, ax, color="k", width=3)

        x_line, = ax.plot([0, 1], [0, 0], [0, 0], lw=2, c="red")
        y_line, = ax.plot([0, 0], [0, 1], [0, 0], lw=2, c="green")
        z_line, = ax.plot([0, 0], [0, 0], [0, 1], lw=2, c="blue")

        plot_data_x = []
        plot_data_y = []
        plot_data_z = []

        def init():
            line.set_data(np.array(plot_data_x), np.array(plot_data_y))
            line.set_3d_properties(np.array(plot_data_z))

            x_line.set_data(np.array([0, 1]), np.array([0, 0]))
            x_line.set_3d_properties(np.array([0, 0]))

            y_line.set_data(np.array([0, 0]), np.array([0, 1]))
            y_line.set_3d_properties(np.array([0, 0]))

            z_line.set_data(np.array([0, 0]), np.array([0, 0]))
            z_line.set_3d_properties(np.array([0, 1]))
            return line, x_line, y_line, z_line

        def animate(i):
            if i >= len(trajectory_data.index):
                return line, x_line, y_line, z_line

            pos = trajectory_data[["position_x [m]", "position_y [m]", "position_z [m]"]].iloc[i].values
            plot_data_x.append(pos[0])
            plot_data_y.append(pos[1])
            plot_data_z.append(pos[2])
            line.set_data(np.array(plot_data_x[-40:]), np.array(plot_data_y[-40:]))
            line.set_3d_properties(np.array(plot_data_z[-40:]))

            rot = trajectory_data[["rotation_x [quaternion]", "rotation_y [quaternion]",
                                   "rotation_z [quaternion]", "rotation_w [quaternion]"]].iloc[i].values
            pos_end = pos + Rotation.from_quat(rot).apply(np.array([1, 0, 0]) * 2)
            x_line.set_data(np.array([pos[0], pos_end[0]]), np.array([pos[1], pos_end[1]]))
            x_line.set_3d_properties(np.array([pos[2], pos_end[2]]))

            pos_end = pos + Rotation.from_quat(rot).apply(np.array([0, 1, 0]) * 2)
            y_line.set_data(np.array([pos[0], pos_end[0]]), np.array([pos[1], pos_end[1]]))
            y_line.set_3d_properties(np.array([pos[2], pos_end[2]]))

            pos_end = pos + Rotation.from_quat(rot).apply(np.array([0, 0, 1]) * 2)
            z_line.set_data(np.array([pos[0], pos_end[0]]), np.array([pos[1], pos_end[1]]))
            z_line.set_3d_properties(np.array([pos[2], pos_end[2]]))

            return line, x_line, y_line, z_line

        anim = FuncAnimation(fig, animate, init_func=init, frames=len(trajectory_data.index) + 50,
                             interval=int(time_stamp_diff * 1000 * skip), blit=True)
        anim.save(os.path.join(args.output_path, "trajectory_anim{}.mp4".format(
            "_axes_on" if args.anim_grid else "")), writer="ffmpeg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay trajectories", fromfile_prefix_chars="@")
    parser.add_argument("-tp", "--trajectory_path", type=str, required=True, help="Path to trajectory/trajectories")
    parser.add_argument("-op", "--output_path", type=str, required=True)
    parser.add_argument("-pp", "--pub_port", type=int, default=10253, help="Flightmare publisher port")
    parser.add_argument("-sp", "--sub_port", type=int, default=10254, help="Flightmare subscriber port")
    parser.add_argument("-o", "--outputs", type=str, nargs="+", default=["anim"], choices=["flightmare", "anim"])
    parser.add_argument("-ag", "--anim_grid", action="store_true")

    arguments = parser.parse_args()
    main(arguments)
