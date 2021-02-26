import os
import numpy as np
import pandas as pd
import argparse
import time
import shutil

from dda.simulation import FlightmareSimulation
from dda.learning import ControllerLearning
from dda.config.settings import create_settings


def save_trajectory_data(time_stamps, mpc_actions, network_actions, states,
                         network_used, max_time, switch_time, repetition, save_path):
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
        "velocity_x [m/s]": states[:, 7],
        "velocity_y [m/s]": states[:, 8],
        "velocity_z [m/s]": states[:, 9],
        "omega_x [rad/s]": states[:, 10],
        "omega_y [rad/s]": states[:, 11],
        "omega_z [rad/s]": states[:, 12],
        "network_used": np.array(network_used).astype(int)
    }
    data = pd.DataFrame(data)
    data.to_csv(os.path.join(save_path, "mpc2nw_mt-{:02d}_st-{:02d}_{:02d}.csv"
                             .format(int(max_time * 10), int(switch_time * 10), repetition)), index=False)


def find_paths(model_load_path, trajectory_path):
    # it is assumed that the file structure is the same as that which DDA creates
    # root directory (to save everything in as well)
    root_dir = os.path.abspath(os.path.join(model_load_path, os.pardir, os.pardir))

    # get the actual file to load from
    model_load_path_no_ext = os.path.splitext(model_load_path)[0]

    # settings file
    settings_file = None
    for file in os.listdir(root_dir):
        if file.endswith(".yaml"):
            settings_file = os.path.join(root_dir, file)
            break

    # save dir for the test trajectories
    model_name = os.path.basename(settings_file)
    model_name = model_name.split(".")[0].replace("snaga_", "")
    save_dir = os.path.join(root_dir, f"dda_{model_name}")

    # figure out whether it is a single trajectory or multiple
    trajectory_paths = []
    if os.path.isfile(trajectory_path) and trajectory_path.endswith(".csv"):
        trajectory_paths.append(os.path.abspath(trajectory_path))
    elif os.path.isdir(trajectory_path):
        for file in os.listdir(trajectory_path):
            if file.startswith("trajectory") and file.endswith(".csv"):
                trajectory_paths.append(os.path.abspath(os.path.join(trajectory_path, file)))
    else:
        raise FileNotFoundError("Path '{}' is not a valid trajectory file or folder".format(trajectory_path))

    return root_dir, model_load_path_no_ext, settings_file, save_dir, trajectory_paths


def main(args):
    root_dir, model_load_path_no_ext, settings_file, save_dir, trajectory_paths = find_paths(
        args.model_load_path, args.trajectory_path)

    # create the directory to save the outputs in if it doesn't exist already
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # copy the settings file to the save directory
    shutil.copy(settings_file, save_dir)

    # create and modify settings
    settings = create_settings(settings_file, mode="dagger", generate_log=False)
    settings.resume_training = True
    settings.resume_ckpt_file = model_load_path_no_ext
    settings.gpu = args.gpu
    settings.flightmare_pub_port = args.pub_port
    settings.flightmare_sub_port = args.sub_port
    settings.max_time = 30.0

    # create simulation (do it here so we don't disconnect and mess things up on snaga)
    simulation = FlightmareSimulation(settings, trajectory_paths[0], max_time=settings.max_time)

    # connect to the simulation either at the start or after training has been run
    simulation.connect_unity(settings.flightmare_pub_port, settings.flightmare_sub_port)

    # using "learner" as controller
    controller = ControllerLearning(settings, trajectory_paths[0], mode="testing", max_time=settings.max_time)

    # wait until Unity rendering/image queue has calmed down
    for _ in range(50):
        simulation.flightmare_wrapper.get_image()
        time.sleep(0.1)

    # test for each specified trajectory:
    for trajectory_path in trajectory_paths:
        trajectory_start = time.time()
        print("\n[Testing] Starting testing for '{}'\n".format(trajectory_path))

        # determine the directory to save the output in
        trajectory_name = os.path.basename(trajectory_path)
        trajectory_name = trajectory_name.split(".")[0]
        trajectory_dir = os.path.join(save_dir, trajectory_name)
        if not os.path.exists(trajectory_dir):
            os.makedirs(trajectory_dir)

        # copy the original trajectory file to that folder for reference
        shutil.copyfile(trajectory_path, os.path.join(trajectory_dir, "original.csv"))

        # update the simulation and learner, which contain trajectory samplers/planners
        simulation.update_trajectory(trajectory_path, max_time=settings.max_time)
        controller.update_trajectory(trajectory_path, max_time=settings.max_time)

        # repeatedly fly the current trajectory
        for repetition in range(args.repetitions):
            repetition_start = time.time()
            print("\n[Testing] Starting repetition {}\n".format(repetition))

            # data to record
            time_stamps = []
            states = []
            mpc_actions = []
            network_actions = []
            network_used = []

            # whether to use the network instead of the MPC
            use_network = False

            # resetting everything
            trajectory_done = False
            info_dict = simulation.reset()
            controller.reset()
            controller.use_network = use_network
            controller.record_data = False
            controller.update_info(info_dict)
            controller.prepare_network_command()
            controller.prepare_expert_command()
            action = controller.get_control_command()

            # run the main loop until the simulation "signals" that the trajectory is done
            while not trajectory_done:
                # decide whether to switch to network at the current time
                if info_dict["time"] > settings.start_buffer:
                    use_network = True
                    controller.use_network = use_network

                # print(info_dict["reference"][:3])

                # record states
                time_stamps.append(info_dict["time"])
                states.append(info_dict["state"])

                # record actions
                mpc_actions.append(action["expert"] if not use_network or args.record_mpc_actions
                                   else np.array([np.nan] * 4))
                network_actions.append(action["network"])
                network_used.append(action["use_network"])

                # perform the step(s) in the simulation and get the new action
                info_dict = simulation.step(action["network"] if action["use_network"] else action["expert"])
                if use_network and not args.record_mpc_actions:
                    info_dict["update"]["expert"] = False

                trajectory_done = info_dict["done"]
                if not trajectory_done:
                    controller.update_info(info_dict)
                    if not settings.save_at_net_frequency or info_dict["update"]["command"]:
                        action = controller.get_control_command()

            # prepare data
            states = np.vstack(states)
            mpc_actions = np.vstack(mpc_actions)
            network_actions = np.vstack(network_actions)

            # save the data
            save_trajectory_data(time_stamps, mpc_actions, network_actions, states, network_used, settings.max_time,
                                 settings.start_buffer, repetition, trajectory_dir)

            print("\n[Testing] Finished repetition {} in {:.2f}s\n".format(repetition, time.time() - repetition_start))

        print("\n[Testing] Finished testing for '{}' in {:.2f}s\n".format(trajectory_path, time.time() - trajectory_start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Network")
    parser.add_argument("-mlp", "--model_load_path", type=str, help="Path to model checkpoint", required=True)
    parser.add_argument("-tp", "--trajectory_path", type=str, help="Path to trajectory/trajectories", required=True)
    parser.add_argument("-rep", "--repetitions", type=int, default=20, help="Repetitions for testing")
    parser.add_argument("-pp", "--pub_port", type=int, default=10253, help="Flightmare publisher port")
    parser.add_argument("-sp", "--sub_port", type=int, default=10254, help="Flightmare subscriber port")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="Flightmare subscriber port")
    parser.add_argument("-rma", "--record_mpc_actions", action="store_true", help="Whether or not to do this")

    main(parser.parse_args())
