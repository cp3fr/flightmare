# TODO: use the DDA simulation only with an MPC
# should be able to specify a trajectory file
# should record the tracking error and collisions (very similar to DDA loop)
# should be able to specify the quadrotor parameters and then...
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

from pathlib import Path
from time import sleep

from planning.planner import TrajectoryPlanner
from planning.mpc_solver import MPCSolver
from dda.simulation import FlightmareSimulation


class SimpleController:

    def __init__(self, trajectory_path, planning_time_horizon, planning_time_step):
        self.planning_time_horizon = planning_time_horizon
        self.planning_time_step = planning_time_step

        self.planner = TrajectoryPlanner(trajectory_path, planning_time_horizon, planning_time_step)
        self.controller = MPCSolver(planning_time_horizon, planning_time_step)

        self.simulation_time = 0.0
        self.state = None
        self.control_command = None

        self.total_solve_time = 0.0
        self.total_solve_count = 0

    def prepare_command(self):
        start = time.time()

        # get the reference trajectory over the time horizon
        planned_traj = self.planner.plan(self.state[:10], self.simulation_time)
        planned_traj = np.array(planned_traj)

        # run non-linear model predictive control
        optimal_action, predicted_traj, cost = self.controller.solve(planned_traj)
        self.control_command = optimal_action

        self.total_solve_time += time.time() - start
        self.total_solve_count += 1

        return self.control_command

    def update(self, info_dict):
        self.simulation_time = info_dict["time"]
        self.state = info_dict["state"][:13]
        if info_dict["update"]["command"]:
            self.prepare_command()

    def get_control_command(self):
        return self.control_command

    def get_stats(self):
        return {
            "total_solve_time": self.total_solve_time,
            "total_solve_count": self.total_solve_count,
            "mean_solve_time": self.total_solve_time / self.total_solve_count
        }


def create_dummy_simulation_settings():
    class Dummy:
        def __init__(self):
            self.env_config_path = ""
            self.use_raw_imu_data = False
            self.base_frequency = 100.0
            self.image_frequency = 15.0
            self.ref_frequency = 50.0
            self.command_frequency = 25.0
            self.expert_command_frequency = 20.0

    settings = Dummy()
    return settings


def format_trajectory(time_stamps, states, references, actions, collisions):
    data = {
        "t": time_stamps,
        "px": states[:, 0],
        "py": states[:, 1],
        "pz": states[:, 2],
        "qw": states[:, 3],
        "qx": states[:, 4],
        "qy": states[:, 5],
        "qz": states[:, 6],
        "vx": states[:, 7],
        "vy": states[:, 8],
        "vz": states[:, 9],
        "wx": states[:, 10],
        "wy": states[:, 11],
        "wz": states[:, 12],
        "px_ref": references[:, 0],
        "py_ref": references[:, 1],
        "pz_ref": references[:, 2],
        "qw_ref": references[:, 3],
        "qx_ref": references[:, 4],
        "qy_ref": references[:, 5],
        "qz_ref": references[:, 6],
        "vx_ref": references[:, 7],
        "vy_ref": references[:, 8],
        "vz_ref": references[:, 9],
        "wx_ref": references[:, 10],
        "wy_ref": references[:, 11],
        "wz_ref": references[:, 12],
        "throttle": actions[:, 0],
        "roll": actions[:, 1],
        "pitch": actions[:, 2],
        "yaw": actions[:, 3],
        "collisions": collisions,
    }
    data = pd.DataFrame(data)
    return data


def run(args):
    # hard-coded for now, should maybe be parameters
    planning_time_horizon = 2.0
    planning_num_steps = 20
    planning_time_step = planning_time_horizon / planning_num_steps

    num_error_correspondences = 30

    # prepare paths
    filename = "mpc_tracking_{:03d}_{:03d}".format(int(planning_time_horizon * 100), int(planning_time_step * 100))
    output_path = Path(args.output_path) / Path(args.trajectory_path).stem
    output_path.mkdir(exist_ok=True)

    # create planner and MPController
    controller = SimpleController(args.trajectory_path, planning_time_horizon, planning_time_step)

    # create Flightmare simulation
    settings = create_dummy_simulation_settings()
    simulation = FlightmareSimulation(settings, args.trajectory_path)

    # create video writer if that's the plan
    writer = None
    if args.save_video:
        writer = cv2.VideoWriter(
            (output_path / filename).with_suffix(".mp4").as_posix(),
            cv2.VideoWriter_fourcc("m", "p", "4", "v"),
            settings.image_frequency,
            (800, 600),
            True,
        )

    # connect simulation
    simulation.connect_unity(args.pub_port, args.sub_port)
    for _ in range(50):
        simulation.flightmare_wrapper.render()
        simulation.flightmare_wrapper.get_image()
        sleep(0.1)

    # initialize everything
    info_dict = simulation.reset()
    controller.update(info_dict)
    action = controller.get_control_command()

    if args.save_video:
        writer.write(info_dict["image"])

    # record some data
    timestamps = []
    references = []
    states = []
    actions = []
    collisions = []

    timestamps.append(info_dict["time"])
    references.append(info_dict["reference"])
    states.append(info_dict["state"][:13])
    collisions.append(info_dict["collision"])
    actions.append(action)

    trajectory_done = False
    while not trajectory_done:
        # run forward the simulation/physics
        info_dict = simulation.step(action)
        trajectory_done = info_dict["done"]

        if args.save_video:
            writer.write(info_dict["image"])

        timestamps.append(info_dict["time"])
        references.append(info_dict["reference"])
        states.append(info_dict["state"][:13])
        collisions.append(info_dict["collision"])
        actions.append(action)

        """
        # I think this was mostly here, because it crashed the collision detector
        if info_dict["collision"]:
            trajectory_done = True
        """

        if not trajectory_done:
            # update the controller with the results
            controller.update(info_dict)

            # get the new action
            action = controller.get_control_command()

    simulation.disconnect_unity()

    if args.save_video:
        writer.release()

    timestamps = np.array(timestamps)
    references = np.array(references)
    states = np.array(states)
    collisions = np.array(collisions)
    actions = np.array(actions)

    # save data
    data = format_trajectory(timestamps, states, references, actions, collisions)
    data.to_csv((output_path / filename).with_suffix(".csv"), index=False)

    # print trajectory error
    trajectory_error = np.mean(np.linalg.norm(references[..., :3] - states[..., :3], axis=1))
    print("Trajectory error | #collisions for h = {}, s = {}: {} | {}".format(
        planning_time_horizon, planning_num_steps, trajectory_error, np.sum(collisions)))

    stats = controller.get_stats()
    print("Average solve time per command: {}s".format(stats["mean_solve_time"]))

    # plot
    plot_position_error(data, (output_path / (filename + "_pos")).with_suffix(".png").as_posix(),
                        planning_time_horizon, planning_time_step)
    plot_rotation_error(data, (output_path / (filename + "_rot")).with_suffix(".png").as_posix(),
                        planning_time_horizon, planning_time_step)


def plot(args):
    # prepare paths
    data_path = Path(args.trajectory_path)

    # TODO: parse info if possible

    # load the data
    data = pd.read_csv(data_path)

    # plot it
    plot_position_error(data, (data_path.parent / (data_path.stem + "_pos")).with_suffix(".png").as_posix())
    plot_rotation_error(data, (data_path.parent / (data_path.stem + "_rot")).with_suffix(".png").as_posix())


def plot_position_error(data, save_path=None, planning_time_horizon=None,
                        planning_time_step=None, num_error_correspondences=30):
    # get error correspondences
    error_correspondences = np.arange(0, len(data.index))
    error_correspondences = error_correspondences[::(max(1, len(data.index) // num_error_correspondences))]

    # plot the two trajectories
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), dpi=100)

    ax[0].plot(data["px_ref"], data["py_ref"], label="reference", color="#00a32e", linewidth=3)
    for idx in error_correspondences:
        ax[0].plot([data["px_ref"].iloc[idx], data["px"].iloc[idx]], [data["py_ref"].iloc[idx], data["py"].iloc[idx]],
                   color="r", label="error", marker="x", linewidth=2)
    ax[0].plot(data["px"], data["py"], label="mpc", color="#03a7ff")
    ax[0].set_xlabel("x-position")
    ax[0].set_ylabel("y-position")
    handles, labels = ax[0].get_legend_handles_labels()
    handles, labels = handles[:2] + handles[-1:], labels[:2] + labels[-1:],
    ax[0].legend(handles, labels)

    ax[1].plot(data["px_ref"], data["pz_ref"], label="reference", color="#00a32e", linewidth=3)
    for idx in error_correspondences:
        ax[1].plot([data["px_ref"].iloc[idx], data["px"].iloc[idx]], [data["pz_ref"].iloc[idx], data["pz"].iloc[idx]],
                   color="r", label="error", marker="x", linewidth=2)
    ax[1].plot(data["px"], data["pz"], label="mpc", color="#03a7ff")
    ax[1].set_xlabel("x-position")
    ax[1].set_ylabel("z-position")
    handles, labels = ax[1].get_legend_handles_labels()
    handles, labels = handles[:2] + handles[-1:], labels[:2] + labels[-1:],
    ax[1].legend(handles, labels)

    fig.suptitle("MPC position tracking{}{}{}".format(
        " for" if planning_time_horizon is not None or planning_time_step is not None else "",
        f" h = {planning_time_horizon}" if planning_time_horizon is not None else "",
        f" s = {planning_time_step}" if planning_time_step is not None else "",
    ))
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_rotation_error(data, save_path=None, planning_time_horizon=None,
                        planning_time_step=None, num_error_correspondences=30):
    # get error correspondences
    error_correspondences = np.arange(0, len(data.index))
    error_correspondences = error_correspondences[::(max(1, len(data.index) // num_error_correspondences))]

    # plot the two trajectories
    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(12, 8), dpi=100)

    for row_idx, name in enumerate(["qx", "qy", "qz", "qw"]):
        ax[row_idx].plot(data["t"], data[f"{name}_ref"], label="reference", color="#00a32e", linewidth=3)
        for idx in error_correspondences:
            ax[row_idx].plot([data["t"].iloc[idx], data["t"].iloc[idx]],
                             [data[f"{name}_ref"].iloc[idx], data[name].iloc[idx]],
                             color="r", label="error", marker="x", linewidth=2)
        ax[row_idx].plot(data["t"], data[name], label="mpc", color="#03a7ff")
        ax[row_idx].set_xlabel("time")
        ax[row_idx].set_ylabel(name)
        handles, labels = ax[0].get_legend_handles_labels()
        handles, labels = handles[:2] + handles[-1:], labels[:2] + labels[-1:],
        ax[row_idx].legend(handles, labels)

    fig.suptitle("MPC rotation tracking{}{}{}".format(
        " for" if planning_time_horizon is not None or planning_time_step is not None else "",
        f" h = {planning_time_horizon}" if planning_time_horizon is not None else "",
        f" s = {planning_time_step}" if planning_time_step is not None else "",
    ))
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test MPC")
    parser.add_argument("-md", "--mode", type=str, default="run", choices=["run", "plot"])
    parser.add_argument("-tp", "--trajectory_path", type=str, help="Path to trajectory/trajectories")
    parser.add_argument("-rp", "--results_path", type=str, help="Path to previous results for plotting")
    parser.add_argument("-op", "--output_path", type=str, help="Output folder")
    parser.add_argument("-pp", "--pub_port", type=int, default=10253, help="Flightmare publisher port")
    parser.add_argument("-sp", "--sub_port", type=int, default=10254, help="Flightmare subscriber port")
    parser.add_argument("-sv", "--save_video", action="store_true",
                        help="Whether or not to save the frames as a video.")

    args_ = parser.parse_args()
    if args_.mode == "run":
        run(args_)
    elif args_.mode == "plot":
        plot(args_)
