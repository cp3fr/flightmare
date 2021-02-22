#!/usr/bin/env python3

import argparse
import os
import csv
import time

import numpy as np
from dda.config.settings import create_settings

from dda.simulation import FlightmareSimulation
from dda.learning import ControllerLearning
from test_networks_tf import show_state_action_plots, save_trajectory_data


class Trainer:

    def __init__(self, settings):
        self.settings = settings

        # TODO: should probably use more than just 1 trajectory for learning and especially for testing
        #  => see whether controller for one trajectory generalises well to another
        self.trajectory_path = self.settings.trajectory_path
        self.trajectory_done = False

        self.simulation = FlightmareSimulation(self.settings, self.trajectory_path,
                                               max_time=self.settings.max_time)
        self.learner = ControllerLearning(self.settings, self.trajectory_path,
                                          mode="iterative", max_time=self.settings.max_time)
        self.connect_to_sim = True

    def perform_testing(self):
        print("\n-----------------------------------------")
        print("[Trainer] Starting Testing (rollout {: >4d})".format(self.learner.rollout_idx))
        print("-----------------------------------------\n")

        testing_start = time.time()

        # defining some settings
        max_time = min(self.settings.max_time + 4.0, self.simulation.reference_sampler.get_final_time_stamp())
        switch_times = np.array(np.arange(0.0, max_time - 2.0, step=2.0).tolist() + [max_time + 1.0])
        switch_times += self.settings.start_buffer
        repetitions = 1
        save_path = os.path.join(self.settings.log_dir, "online_eval_rollout-{:04d}".format(self.learner.rollout_idx))
        os.makedirs(save_path)

        # connect to sim if testing happens after training
        if self.connect_to_sim:
            self.simulation.connect_unity(self.settings.flightmare_pub_port, self.settings.flightmare_sub_port)
            self.connect_to_sim = False

            # wait until Unity rendering/image queue is actually ready
            for _ in range(20):
                self.simulation.flightmare_wrapper.get_image()
                time.sleep(0.1)

        for switch_time in switch_times:
            for repetition in range(repetitions if switch_time < max_time else 1):
                print("\n[Trainer] Testing for switch time {}, repetition {}\n".format(switch_time, repetition))

                repetition_start = time.time()

                # data to record
                states = []
                reduced_states = []
                mpc_actions = []
                network_actions = []
                network_used = []
                time_stamps = []

                # whether to use the network instead of the MPC
                use_network = False

                # resetting everything
                trajectory_done = False
                info_dict = self.simulation.reset()
                self.learner.reset(new_rollout=False)
                self.learner.mode = "testing"
                self.learner.use_network = use_network
                self.learner.record_data = False
                self.learner.update_info(info_dict)
                self.learner.prepare_expert_command()
                action = self.learner.get_control_command()

                # run the main loop until the simulation "signals" that the trajectory is done
                while not trajectory_done:
                    if info_dict["time"] > switch_time:
                        use_network = True
                        self.learner.use_network = use_network

                    # record states first
                    time_stamps.append(info_dict["time"])
                    states.append(info_dict["state"])
                    reduced_states.append(info_dict["state"][:10])

                    # record actions after the decision has been made for the current state
                    mpc_actions.append(action["expert"])
                    network_actions.append(action["network"])
                    network_used.append(action["use_network"])

                    info_dict = self.simulation.step(action["network"] if action["use_network"] else action["expert"])
                    trajectory_done = info_dict["done"]
                    if not trajectory_done:
                        self.learner.update_info(info_dict)
                        action = self.learner.get_control_command()

                states = np.vstack(states)
                reduced_states = np.vstack(reduced_states)
                mpc_actions = np.vstack(mpc_actions)
                network_actions = np.vstack(network_actions)

                """
                show_state_action_plots(self.trajectory_path, reduced_states, mpc_actions, network_actions,
                                        self.simulation.base_time_step, self.simulation.total_time,
                                        save_path="")
                """
                save_trajectory_data(time_stamps, mpc_actions, network_actions, states,
                                     network_used, max_time, switch_time, repetition, save_path)

                print("\n[Trainer] Finished testing for switch time {:.1f}s, repetition {} after {:.2f}s\n"
                      .format(switch_time, repetition, time.time() - repetition_start))

        self.learner.mode = "iterative"

        print("\n--------------------------------------------------------")
        print("[Trainer] Finished testing after {:.2f}s (rollout {})"
              .format(time.time() - testing_start, self.learner.rollout_idx))
        print("--------------------------------------------------------\n")

    def perform_training(self):
        shutdown_requested = False
        self.connect_to_sim = True

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

        with open(os.path.join(self.settings.log_dir, "metrics.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerows([["rollout", "traj_err_mean", "traj_err_median",
                               "expert_usage", "network_usage", "randomised_usage"]])

        while (not shutdown_requested) and (self.learner.rollout_idx < self.settings.max_rollouts):
            rollout_start = time.time()

            self.trajectory_done = False
            info_dict = self.simulation.reset()
            if info_dict["time"] > self.settings.start_buffer:
                self.learner.start_data_recording()
            self.learner.reset()
            self.learner.update_info(info_dict)
            self.learner.prepare_network_command()
            self.learner.prepare_expert_command()
            action = self.learner.get_control_command()

            # connect to the simulation either at the start or after training has been run
            if self.connect_to_sim:
                self.simulation.connect_unity(self.settings.flightmare_pub_port, self.settings.flightmare_sub_port)
                self.connect_to_sim = False

                # wait until Unity rendering/image queue is actually ready
                for _ in range(50):
                    self.simulation.flightmare_wrapper.get_image()
                    time.sleep(0.1)

            # keep track of some metrics
            ref_log = []
            gt_pos_log = []
            error_log = []
            # TODO: maybe only collect this when data is being recorded
            # states = []

            # run the main loop until the simulation "signals" that the trajectory is done
            print("\n[Trainer] Starting experiment {}\n".format(self.learner.rollout_idx))
            step_counter = 0
            while not self.trajectory_done:
                # TODO?: whenever the image has been update, get the feature tracks and visualise them in a video
                #  to check whether everything with the image capturing and feature tracking works correctly
                info_dict = self.simulation.step(action["network"] if action["use_network"] else action["expert"])
                self.trajectory_done = info_dict["done"]
                if not self.trajectory_done:
                    if info_dict["time"] > self.settings.start_buffer and not self.learner.record_data:
                        self.learner.start_data_recording()

                    self.learner.update_info(info_dict)
                    action = self.learner.get_control_command()

                    if self.learner.record_data:
                        pos_ref_dict = self.learner.compute_trajectory_error()
                        gt_pos_log.append(pos_ref_dict["gt_pos"])
                        ref_log.append(pos_ref_dict["gt_ref"])
                        error_log.append(np.linalg.norm(pos_ref_dict["gt_pos"] - pos_ref_dict["gt_ref"]))
                step_counter += 1

            # final logging
            tracking_error = np.mean(error_log)
            median_traj_error = np.median(error_log)
            t_log = np.stack((ref_log, gt_pos_log), axis=0)
            usage = self.learner.stop_data_recording()

            rollout_time = time.time() - rollout_start
            print("\n[Trainer]")
            print("Finished rollout after {:.2f}s ({:.4f}s average per step)"
                  .format(rollout_time, rollout_time / step_counter))
            print("Expert|network|randomised used {:06.3f}|{:06.3f}|{:06.3f}% of the time"
                  .format(100.0 * usage["expert"], 100.0 * usage["network"], 100.0 * usage["randomised"]))
            print("Mean tracking error is {:.3f}".format(tracking_error))
            print("Median tracking error is {:.3f}\n".format(median_traj_error))

            with open(os.path.join(self.settings.log_dir, "metrics.csv"), "a") as f:
                writer = csv.writer(f)
                writer.writerows([[self.learner.rollout_idx, tracking_error, median_traj_error,
                                   usage["expert"], usage["network"], usage["randomised"]]])

            if self.learner.rollout_idx % self.settings.train_every_n_rollouts == 0:
                # here the simulation basically seems to be stopped while the network is being trained
                if self.settings.disconnect_when_training:
                    self.simulation.disconnect_unity()
                    self.connect_to_sim = True
                self.learner.train()

            if self.learner.rollout_idx % self.settings.test_every_n_rollouts == 0:
                self.perform_testing()

            if (self.learner.rollout_idx % self.settings.double_th_every_n_rollouts) == 0:
                # this seems to encourage more use of the network/more exploration the further we progress in training
                print("\n[Trainer]")
                self.settings.fallback_threshold_rates += 0.5
                print("Setting rate threshold to {: >2.2f}".format(self.settings.fallback_threshold_rates))
                self.settings.rand_controller_prob = np.minimum(0.3, self.settings.rand_controller_prob * 2)
                print("Setting rand controller probability to {: >2.2f}%\n".format(self.settings.rand_controller_prob * 100.0))

            if self.settings.verbose:
                t_log_fname = os.path.join(self.settings.log_dir, "traj_log_{:5d}.npy".format(self.learner.rollout_idx))
                np.save(t_log_fname, t_log)

            """
            states = np.vstack(states)
            show_state_action_plots(self.settings.trajectory_path, states, None, None,
                                    self.simulation.command_time_step, self.simulation.total_time)
            break
            """
            # break


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
