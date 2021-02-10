#!/usr/bin/env python3

import argparse
import os
import time
import cv2

import numpy as np
from dda.config.settings import create_settings

from dda.simulation import FlightmareSimulation
from dda.learning import ControllerLearning


class Trainer:

    def __init__(self, settings):
        self.settings = settings
        self.trajectory_done = False

        # TODO: these should probably also be in the settings file
        self.base_frequency = 60.0  # should probably be max of the others
        self.state_frequency = 60.0  # (replacement for now for IMU frequency)
        self.image_frequency = 30.0
        self.command_frequency = 20.0  # should this be as high as it is in DDA originally?

        self.base_time_step = 1.0 / self.base_frequency
        self.state_time_step = 1.0 / self.state_frequency
        self.image_time_step = 1.0 / self.image_frequency
        self.command_time_step = 1.0 / self.command_frequency

        # TODO: should probably use more than just 1 trajectory for learning and especially for testing
        #  => see whether controller for one trajectory generalises well to another
        self.trajectory_path = "/home/simon/Downloads/trajectory_s016_r05_flat_li01.csv"

        # TODO: want only access to simulation and Unity interface here => might need to restructure stuff
        #  so that expert is not actually contained in the simulation => either that or keep things as they are
        # ideally, would probably like to have one simulation object that also interfaces with Unity in whatever way
        # is required at the moment (could e.g. use Flightmare dynamics instead of Python for integrating)

    def perform_training(self):
        shutdown_requested = False
        connect_to_sim = True

        trajectory_path = "/home/simon/Downloads/trajectory_s016_r05_flat_li01.csv"
        max_time = 3.0
        simulation = FlightmareSimulation(self.settings, trajectory_path, max_time=max_time)
        learner = ControllerLearning(self.settings, trajectory_path, mode="iterative", max_time=max_time)

        if self.settings.execute_nw_predictions:
            print("\n-------------------------------------------")
            print("[Trainer]")
            print("Running Dagger with the following params")
            print("Rates threshold: {}; Rand Controller Th {}".format(self.settings.fallback_threshold_rates,
                                                                      self.settings.rand_controller_prob))
            print("-------------------------------------------\n")
        else:
            print("\n---------------------------")
            print("[Trainer] Collecting Data with Expert")
            print("---------------------------\n")

        while (not shutdown_requested) and (learner.rollout_idx < self.settings.max_rollouts):
            # TODO: add switch into training mode when MPC is started earlier to reach "stable" flight

            self.trajectory_done = False
            info_dict = simulation.reset()
            learner.start_data_recording()
            learner.reset()
            learner.update_info(info_dict)
            learner.prepare_expert_command()
            action = learner.get_control_command()

            # connect to the simulation either at the start or after training has been run
            if connect_to_sim:
                simulation.connect_unity(self.settings.flightmare_pub_port, self.settings.flightmare_sub_port)
                connect_to_sim = False

            """
            # will have to see whether writing videos for individual runs should be added again at 
            # some point; it is more likely that this will only be done for testing the networks
            
            writer = cv2.VideoWriter(
                # "/home/simon/Desktop/flightmare_cam_test/alphapilot_arena_mpc_async_5.mp4",
                # "/home/simon/Desktop/weekly_meeting/meeting14/cam_angle_test_mpc_wave_fast_rot.mp4",
                "/home/simon/Desktop/flightmare_cam_test/new_sim_test.mp4",
                cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                float(simulation.image_frequency),
                (simulation.flightmare_wrapper.image_width, simulation.flightmare_wrapper.image_height),
                True
            )
            """

            # keep track of some metrics
            ref_log = []
            gt_pos_log = []
            error_log = []

            # run the main loop until the simulation "signals" that the trajectory is done
            print("\n[Trainer] Starting experiment {}\n".format(learner.rollout_idx))
            while not self.trajectory_done:
                info_dict, successes = simulation.step(action)
                self.trajectory_done = info_dict["done"]
                if not self.trajectory_done:
                    learner.update_info(info_dict)
                    action = learner.get_control_command()

                    pos_ref_dict = learner.compute_trajectory_error()
                    gt_pos_log.append(pos_ref_dict["gt_pos"])
                    ref_log.append(pos_ref_dict["gt_ref"])
                    error_log.append(np.linalg.norm(pos_ref_dict["gt_pos"] - pos_ref_dict["gt_ref"]))

                """
                writer.write(info_dict["image"])
                states.append(info_dict["state"])
                actions.append(action)
                """

            # writer.release()

            """
            states = np.vstack(states)
            actions = np.vstack(actions)

            trajectory = pd.read_csv(trajectory_path)
            trajectory = trajectory[trajectory["time-since-start [s]"] <= simulation.total_time]
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

            print("states", states.shape)
            print("trajectory", trajectory.shape)

            visualise_states(states, trajectory, simulation.total_time, simulation.command_time_step, True)
            visualise_actions(actions, simulation.total_time, simulation.command_time_step, True)
            """

            # final logging
            tracking_error = np.mean(error_log)
            median_traj_error = np.median(error_log)
            t_log = np.stack((ref_log, gt_pos_log), axis=0)
            expert_usage = learner.stop_data_recording()

            print("\n[Trainer]")
            print("Expert used {:.03f}% of the times".format(100.0 * expert_usage))
            print("Mean Tracking Error is {:.03f}".format(tracking_error))
            print("Median Tracking Error is {:.03f}\n".format(median_traj_error))

            if learner.rollout_idx % self.settings.train_every_n_rollouts == 0:
                # here the simulation basically seems to be stopped while the network is being trained
                simulation.disconnect_unity()
                connect_to_sim = True
                learner.train()

            if (learner.rollout_idx % self.settings.double_th_every_n_rollouts) == 0:
                # this seems to encourage more use of the network/more exploration the further we progress in training
                print("\n[Trainer]")
                self.settings.fallback_threshold_rates += 0.5
                print("Setting Rate Threshold to {}".format(self.settings.fallback_threshold_rates))
                self.settings.rand_controller_prob = np.minimum(0.3, self.settings.rand_controller_prob * 2)
                print("Setting Rand Controller Prob to {}\n".format(self.settings.rand_controller_prob))

            if self.settings.verbose:
                t_log_fname = os.path.join(self.settings.log_dir, "traj_log_{:5d}.npy".format(learner.rollout_idx))
                np.save(t_log_fname, t_log)

    def perform_testing(self):
        """
        learner = TrajectoryLearning.TrajectoryLearning(self.settings, mode="testing")
        shutdown_requested = False
        rollout_idx = 0
        while (not shutdown_requested) and (rollout_idx < self.settings.max_rollouts):
            self.trajectory_done = False
            # setup_sim()
            if self.settings.verbose:
                # Will save data for debugging reasons
                learner.start_data_recording()
            # self.start_experiment(learner)
            # start_time = time.time()
            # time_run = 0

            # this loop basically needs to be manual advancing of the simulation etc.
            ref_log = []
            gt_pos_log = []
            error_log = []
            while not self.trajectory_done:  # and (time_run < 100):
                # time.sleep(0.1)
                # time_run = time.time() - start_time
                if learner.use_network and learner.reference_updated:
                    pos_ref_dict = learner.compute_trajectory_error()
                    gt_pos_log.append(pos_ref_dict["gt_pos"])
                    ref_log.append(pos_ref_dict["gt_ref"])
                    error_log.append(np.linalg.norm(pos_ref_dict["gt_pos"] - pos_ref_dict["gt_ref"]))

            # final logging
            tracking_error = np.mean(error_log)
            median_traj_error = np.median(error_log)
            t_log = np.stack((ref_log, gt_pos_log), axis=0)
            expert_usage = learner.stop_data_recording()
            shutdown_requested = learner.shutdown_requested()
            print("{} Rollout: Expert used {:.03f}% of the times".format(rollout_idx + 1, 100.0 * expert_usage))
            print("Mean Tracking Error is {:.03f}".format(tracking_error))
            print("Median Tracking Error is {:.03f}".format(median_traj_error))
            rollout_idx += 1
            if self.settings.verbose:
                t_log_fname = os.path.join(self.settings.log_dir, "traj_log_{:05d}.npy".format(rollout_idx))
                np.save(t_log_fname, t_log)
        """


def main():
    parser = argparse.ArgumentParser(description='Train RAF network.')
    parser.add_argument('--settings_file', help='Path to settings yaml', required=True)

    args = parser.parse_args()
    settings_filepath = args.settings_file
    settings = create_settings(settings_filepath, mode='dagger')
    trainer = Trainer(settings)
    trainer.perform_training()


if __name__ == "__main__":
    main()
