#!/usr/bin/env python3

import argparse
import os
import time
import cv2
import pandas as pd

import numpy as np
# import rospy
# from dda.src.ControllerLearning import TrajectoryLearning
from dda.common import update_mpc_params, setup_sim, initialize_vio
from dda.config.settings import create_settings
# from std_msgs.msg import Bool

from dda.simulation import PythonSimulation, FlightmareSimulation
from dda.learning import ControllerLearning
from run_tests import ensure_quaternion_consistency, visualise_states, visualise_actions


class Trainer:

    def __init__(self, settings):
        # rospy.init_node('iterative_learning_node', anonymous=False)
        self.settings = settings
        self.trajectory_done = False
        # self.traj_done_sub = rospy.Subscriber("/hummingbird/switch_to_network", Bool,
        #                                       self.callback_traj_done, queue_size=1)

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

    def callback_traj_done(self, data):
        self.trajectory_done = data.data

    def start_experiment(self, learner):
        # this entire function only seems to be responsible for initialising everything for VIO, which is used
        # for the expert (I think) => since we only use the GT-state MPC, we can mostly just ignore this

        # reset_success_str = 'rostopic pub /success_reset std_msgs/Empty "{}" -1'
        # os.system(reset_success_str)
        # initialize_vio()

        learner.latest_thrust_factor = 1.0  # Could be changed to adapt for different quadrotors.
        print("Doing experiment {}, with such factor {}".format(learner.rollout_idx, learner.latest_thrust_factor))

        # if True, we will still use the VIO-orientation, even when initialization is poor.
        # If set to False, we will fall back to GT.
        # Set to True only if you are sure your VIO is well calibrated.
        use_chimaera = False  # TODO: basically always set to false, since we use GT
        # check if initialization was good. If not, we will perform rollout with ground truth to not waste time!
        """
        vio_init_good = learner.vio_init_good  # TODO: this should not be needed
        if vio_init_good:
            # rospy.loginfo("VINS-Mono initialization is good, switching to vision-based state estimate!")
            os.system("timeout 1s rostopic pub /switch_odometry std_msgs/Int8 'data: 1'")
        else:
            if use_chimaera:
                rospy.logerr("VINS-Mono initialization is poor, use orientation, bodyrates from VIO and linear velocity estimate from GT!")
                os.system("timeout 1s rostopic pub /switch_odometry std_msgs/Int8 'data: 2'")
            else:
                rospy.logerr("VINS-Mono initialization is poor, keeping ground truth estimate!")
                os.system("timeout 1s rostopic pub /switch_odometry std_msgs/Int8 'data: 0'")
        # Start Flying!
        os.system("timeout 1s rostopic pub /hummingbird/fpv_quad_looping/execute_trajectory std_msgs/Bool 'data: true'")
        """
        return True

    def perform_training(self):
        # learner = TrajectoryLearning.TrajectoryLearning(self.settings, mode="iterative")
        shutdown_requested = False
        connect_to_sim = True
        train_every_n_rollouts = self.settings.train_every_n_rollouts

        trajectory_path = "/home/simon/Downloads/trajectory_s016_r05_flat_li01.csv"
        max_time = 3.0
        # simulation = PythonSimulation(trajectory_path)
        simulation = FlightmareSimulation(trajectory_path, max_time=max_time)
        info_dict = simulation.reset()

        learner = ControllerLearning(self.settings, trajectory_path, mode="iterative", max_time=max_time)

        if self.settings.execute_nw_predictions:
            print("-------------------------------------------")
            print("Running Dagger with the following params")
            print("Rates threshold: {}; Rand Controller Th {}".format(
                self.settings.fallback_threshold_rates,
                self.settings.rand_controller_prob))
            print("-------------------------------------------")
        else:
            print("---------------------------")
            print("Collecting Data with Expert")
            print("---------------------------")

        while (not shutdown_requested) and (learner.rollout_idx < self.settings.max_rollouts):
            """
            # setup_sim()  # pretty sure this just resets the simulation (e.g. to trajectory start position)
            learner.start_data_recording()
            # self.start_experiment(learner)
            print("Starting Experiment {}".format(learner.rollout_idx))
            # start_time = time.time()
            # time_run = 0
            ref_log = []
            gt_pos_log = []
            error_log = []
            # TODO: why are there no explicit calls to advance the simulation etc.?
            #  => probably because all of this should be taken care of by callbacks in the learner;
            #     basically, as soon as the queues/listeners are initialised (and data recording is on etc.),
            #     the recording and control is taken care of elsewhere
            while not self.trajectory_done:  # and (time_run < 100):
                # I guess this is a timeout of 100 seconds for the real world
                # time.sleep(0.1)
                # time_run = time.time() - start_time
                if learner.use_network and learner.reference_updated:
                    pos_ref_dict = learner.compute_trajectory_error()
                    gt_pos_log.append(pos_ref_dict["gt_pos"])
                    ref_log.append(pos_ref_dict["gt_ref"])
                    error_log.append(np.linalg.norm(pos_ref_dict["gt_pos"] - pos_ref_dict["gt_ref"]))
            """

            # TODO:
            #  - main loop (see sim.py)
            #  - maybe just one object that handles the simulation and "holds" the expert/network?
            #    => then it would just have different methods to call at the appropriate times:
            #       - callback_reference (for the trajectory, but what about the expert then? see also command)
            #       - callback_image (for images/feature tracks)
            #       - callback_command (to get the latest command, whether from network or expert)
            #       - callback_state/imu (depending on what works => IMU might also make problems with
            #                             using Python instead of Flightmare "simulation")
            #    => should have the following "main objects"
            #       - python simulation interface
            #       - Unity simulation interface
            #    => actually, should probably have a "controller object" and take care of the simulation outside of that
            #       (similar to the way it's already done in sim.py)

            ref_log = []
            gt_pos_log = []
            error_log = []

            self.trajectory_done = False
            learner.start_data_recording()
            learner.reset()
            learner.update_info(info_dict)
            learner.prepare_expert_command()
            action = learner.get_control_command()

            print("Starting Experiment {}".format(learner.rollout_idx))

            writer = cv2.VideoWriter(
                # "/home/simon/Desktop/flightmare_cam_test/alphapilot_arena_mpc_async_5.mp4",
                # "/home/simon/Desktop/weekly_meeting/meeting14/cam_angle_test_mpc_wave_fast_rot.mp4",
                "/home/simon/Desktop/flightmare_cam_test/new_sim_test.mp4",
                cv2.VideoWriter_fourcc("m", "p", "4", "v"),
                float(simulation.image_frequency),
                (simulation.flightmare_wrapper.image_width, simulation.flightmare_wrapper.image_height),
                True
            )

            if connect_to_sim:
                simulation.connect_unity()
                connect_to_sim = False
            # TODO: probably makes more sense to have this outside and only disconnected when training happens

            states = []
            actions = []
            while not self.trajectory_done:
                info_dict, successes = simulation.step(action)
                self.trajectory_done = info_dict["done"]
                # print(info_dict["state"])

                # print(action)
                if not self.trajectory_done:
                    learner.update_info(info_dict)
                    action = learner.get_control_command()

                    pos_ref_dict = learner.compute_trajectory_error()
                    gt_pos_log.append(pos_ref_dict["gt_pos"])
                    ref_log.append(pos_ref_dict["gt_ref"])
                    error_log.append(np.linalg.norm(pos_ref_dict["gt_pos"] - pos_ref_dict["gt_ref"]))

                    # TODO: same problem of slow expert updates might happen with images
                    #  => actually it is very important that we don't feed the feature tracker the same images
                    #     I think => only when the image is actually updated, should new features be computed
                    #  => only call these methods when stuff has actually been updated? e.g. have list of tuples with
                    #     the first element being a boolean?

                writer.write(info_dict["image"])
                states.append(info_dict["state"])
                actions.append(action)

            writer.release()
            info_dict = simulation.reset()

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

            # TODO: once the simulation loop (rollout) has been changed,
            #  everything else should be mostly plug and play => LMAO, nvm

            # one important thing to note is that TrajectoryLearning seems to mostly be responsible for interfacing
            # with the simulation (including saving the data in the correct format) => that means that it is pretty
            # useless for our purposes (except for code snippets like resampling feature tracks)

            # the other main thing it does it producing control commands and managing whether the expert or the network
            # is "responsible" for generating the commands => not sure whether it is best to separate the "simulation
            # handling" and the "command handling" as I have done in sim.py

            # final logging
            tracking_error = np.mean(error_log)
            median_traj_error = np.median(error_log)
            t_log = np.stack((ref_log, gt_pos_log), axis=0)
            expert_usage = learner.stop_data_recording()

            print("Expert used {:.03f}% of the times".format(100.0 * expert_usage))
            print("Mean Tracking Error is {:.03f}".format(tracking_error))
            print("Median Tracking Error is {:.03f}".format(median_traj_error))
            print("LEARNER ROLLOUT IDX:", learner.rollout_idx)

            if learner.rollout_idx % train_every_n_rollouts == 0:
                # here the simulation basically seems to be stopped while the network is being trained
                simulation.disconnect_unity()
                connect_to_sim = True
                learner.train()  # TODO <======== this is important
            if (learner.rollout_idx % self.settings.double_th_every_n_rollouts) == 0:
                # this seems to encourage more use of the network/more exploration the further we progress in training
                self.settings.fallback_threshold_rates += 0.5
                print("Setting Rate Threshold to {}".format(self.settings.fallback_threshold_rates))
                self.settings.rand_controller_prob = np.minimum(0.3, self.settings.rand_controller_prob * 2)
                print("Setting Rand Controller Prob to {}".format(self.settings.rand_controller_prob))
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
    # update_mpc_params()
    # setup_sim()
    trainer = Trainer(settings)
    trainer.perform_training()


if __name__ == "__main__":
    main()
