import numpy as np
import random
import copy
import os
import csv
import datetime

from collections import deque
from scipy.spatial.transform import Rotation
from dda.src.ControllerLearning.models.bodyrate_learner import BodyrateLearner
from features.feature_tracker import FeatureTracker
from mpc.mpc.mpc_solver import MPCSolver
from mpc.simulation.planner import TrajectoryPlanner

# TODO: not sure why this is the way it is, might have to be adjusted
# e.g. if it stems from most features in the original only being tracked for around 10 steps
# then this normalisation won't do much to bring the value into a similar range as the other
# components of the feature track "features"
TRACK_NUM_NORMALIZE = 10


class ControllerLearning:

    def __init__(self, config, trajectory_path, mode, max_time=None):
        # TODO: trajectory_path should not really be specified like this I think
        #  => we might want to learn using multiple trajectories,
        #     so maybe a planner/sampler should be given instead?

        # "meta" stuff
        self.config = config
        self.mode = mode

        self.csv_filename = None
        self.image_save_dir = None

        # things to keep track of the current "status"
        self.record_data = False
        self.is_training = False
        self.use_network = False
        self.network_initialised = False
        self.reference_updated = False  # this just seems to be a flag to make sure that there is a reference state at all

        self.rollout_idx = 0
        self.n_times_net = 0
        self.n_times_expert = 0
        self.recorded_samples = 0
        self.counter = 0

        self.fts_queue = deque([], maxlen=self.config.seq_len)
        self.state_queue = deque([], maxlen=self.config.seq_len)

        # simulation time (for now mostly for the feature tracker for velocity calculation)
        self.simulation_time = 0.0

        # stuff to keep track of
        self.state = None
        self.state_rot = None
        self.reference = None
        self.reference_rot = None
        self.state_estimate = None
        self.state_estimate_rot = None
        self.feature_tracks = None

        # the current control command, computed either by the expert or the network
        self.control_command = None
        self.network_command = None

        # objects
        self.feature_tracker = FeatureTracker(self.config.min_number_fts * 2)
        self.learner = BodyrateLearner(settings=self.config, expect_partial=(mode == "testing"))
        self.planner = TrajectoryPlanner(trajectory_path, 4.0, 0.2, max_time=max_time)
        self.expert = MPCSolver(4.0, 0.2)

        # preparing for data saving
        if self.mode == "iterative" or self.config.verbose:
            self.write_csv_header()

    def reset(self, new_rollout=True):
        # TODO: if this is used for anything but initialisation,
        #  should also reset feature tracker (and maybe other stuff?)
        self.n_times_net = 0
        self.n_times_expert = 0
        self.counter = 0
        if new_rollout:
            self.rollout_idx += 1

        self.use_network = True

        self.fts_queue.clear()
        self.state_queue.clear()
        # TODO: THESE ARE ACTUALLY PRETTY IMPORTANT => I think the only reason it's set to 1 in dagger_settings.yaml
        #  is to be machine-independent, but it should actually be 8 for the model to be trained

        self.state = np.zeros((13,), dtype=np.float32)
        self.reference = np.zeros((13,), dtype=np.float32)
        self.state_estimate = np.zeros((13,), dtype=np.float32)
        # TODO: these should actually be fewer right? n_init_states should be the number of actual inputs

        if self.config.use_imu:
            if self.config.use_pos:
                n_init_states = 36
            else:
                n_init_states = 30
        else:
            if self.config.use_pos:
                n_init_states = 18
            else:
                n_init_states = 15

        init_dict = {}
        for i in range(self.config.min_number_fts):
            init_dict[i] = np.zeros((5,), dtype=np.float32)

        for _ in range(self.config.seq_len):
            self.state_queue.append(np.zeros((n_init_states,), dtype=np.float32))
            self.fts_queue.append(init_dict)

        self.feature_tracks = copy.copy(init_dict)
        # self.feature_tracks = np.stack([np.stack([v for v in self.fts_queue[j].values()])
        #                                 for j in range(self.config.seq_len)])

    def start_data_recording(self):
        print("[ControllerLearning] Collecting data")
        self.record_data = True

    def stop_data_recording(self):
        print("[ControllerLearning] Stop data collection")
        self.record_data = False
        expert_usage = self.n_times_expert / (self.n_times_net + self.n_times_expert)
        return expert_usage

    def train(self):
        # not sure whether all these booleans are actually relevant
        self.is_training = True
        self.learner.train()
        self.is_training = False
        self.use_network = False  # TODO: probably remove this, since we don't have any VIO init or anything like that
        # TODO: where is the simulation reset though?

    def update_simulation_time(self, simulation_time):
        self.simulation_time = simulation_time

    def update_state(self, state):
        # assumed ordering of state variables is [pos. rot, vel, omega]
        # simulation returns full state (with linear acc and motor torques) => take only first 13 entries
        self.state = state[:13]
        # print("\nState:", self.state, sep="\n")
        self.state_rot = self.state[4:7].tolist() + self.state[3:4].tolist()
        # state_rot_wrong = self.state[3:7]
        # print("State rotation (XYZW):", self.state_rot)
        # print("State rotation wrong (WXYZ):", state_rot_wrong)
        self.state_rot = Rotation.from_quat(self.state_rot).as_matrix()
        # state_rot_wrong = Rotation.from_quat(state_rot_wrong).as_matrix()
        # print("State rotation (matrix):", self.state_rot, sep="\n")
        # print("State rotation wrong (matrix):", state_rot_wrong, sep="\n")
        self.state_rot = self.state_rot.reshape((9,)).tolist()

    def update_reference(self, reference):
        self.reference = reference
        self.reference_rot = self.reference[4:7].tolist() + self.reference[3:4].tolist()
        self.reference_rot = Rotation.from_quat(self.reference_rot).as_matrix().reshape((9,)).tolist()
        if not self.reference_updated:
            self.reference_updated = True

        """
        self.reference = np.array([
            "r_pos_x", "r_pos_y", "r_pos_z",
            "r_rot_w", "r_rot_x", "r_rot_y", "r_rot_z",
            "r_vel_x", "r_vel_y", "r_vel_z",
            "r_omega_x", "r_omega_y", "r_omega_z",
        ])
        self.reference_rot = [
            "r_r00", "r_r01", "r_r02",
            "r_r10", "r_r11", "r_r12",
            "r_r20", "r_r21", "r_r22",
        ]
        """

    def update_state_estimate(self, state_estimate):
        self.state_estimate = state_estimate[:13]
        self.state_estimate_rot = self.state_estimate[4:7].tolist() + self.state_estimate[3:4].tolist()
        self.state_estimate_rot = Rotation.from_quat(self.state_estimate_rot).as_matrix().reshape((9,)).tolist()

        """
        self.state_estimate = np.array([
            "se_pos_x", "se_pos_y", "se_pos_z",
            "se_rot_w", "se_rot_x", "se_rot_y", "se_rot_z",
            "se_vel_x", "se_vel_y", "se_vel_z",
            "se_omega_x", "se_omega_y", "se_omega_z",
        ])
        self.state_estimate_rot = [
            "se_r00", "se_r01", "se_r02",
            "se_r10", "se_r11", "se_r12",
            "se_r20", "se_r21", "se_r22",
        ]
        """

    def update_image(self, image):
        # TODO: for now do the feature track computation/update here, but might want to
        #  consider doing this stuff in a subclass instead (including the stuff in __init__)
        if not self.config.use_fts_tracks and self.mode == "testing":
            return

        # get the features for the current frame
        feature_tracks = self.feature_tracker.process_image(image, current_time=self.simulation_time)

        # "format" the features like original DDA
        features_dict = {}
        for i in range(len(feature_tracks)):
            ft_id = feature_tracks[i][0]
            x = feature_tracks[i][2]
            y = feature_tracks[i][3]
            velocity_x = feature_tracks[i][4]
            velocity_y = feature_tracks[i][5]
            track_count = 2 * (feature_tracks[i][1] / TRACK_NUM_NORMALIZE) - 1  # TODO: probably revise
            feat = np.array([x, y, velocity_x, velocity_y, track_count])
            features_dict[ft_id] = feat

        if len(features_dict.keys()) == 0:
            return

        # remember the "unsampled" features for saving them for training
        self.feature_tracks = copy.copy(features_dict)

        # sample features
        processed_dict = copy.copy(features_dict)
        missing_fts = self.config.min_number_fts - len(features_dict.keys())
        if missing_fts > 0:
            # features are missing
            if missing_fts != self.config.min_number_fts:
                # there is something, we can sample
                new_features_keys = random.choices(list(features_dict.keys()), k=int(missing_fts))
                for j in range(missing_fts):
                    processed_dict[-j - 1] = features_dict[new_features_keys[j]]
            else:
                raise IOError("There should not be zero features!")
        elif missing_fts < 0:
            # there are more features than we need, so sample
            del_features_keys = random.sample(features_dict.keys(), int(-missing_fts))
            for k in del_features_keys:
                del processed_dict[k]

        self.fts_queue.append(processed_dict)
        # self.feature_tracks = np.stack([np.stack([v for v in self.fts_queue[j].values()])
        #                                 for j in range(self.config.seq_len)])
        # TODO: might actually just move this to network input prep, since that's where it's supposed to be used

    def update_info(self, info_dict):
        self.update_simulation_time(info_dict["time"])
        self.update_state(info_dict["state"])
        self.update_state_estimate(info_dict["state_estimate"])
        if info_dict["update"]["image"]:
            # self.update_image(info_dict["image"])
            pass
            # TODO: this needs to be updated to only start doing stuff when the first actual
            #  image arrives, otherwise the feature tracker gets a black image at the start
        if info_dict["update"]["reference"]:
            self.update_reference(info_dict["reference"])
        if info_dict["update"]["expert"]:
            self.prepare_expert_command()
        # TODO: maybe also update with "done" and don't do anything anymore after this if it is

    def prepare_expert_command(self):
        # TODO: need to update expert command thingy
        # get the reference trajectory over the time horizon
        # TODO: should really think about whether the planner isn't the one that should supply the references
        #  => could have the timing be determined by the simulation though
        #  => but then again, we only want to sample the current/next reference state...
        planned_traj = self.planner.plan(self.state[:10], self.simulation_time)
        planned_traj = np.array(planned_traj)

        # run non-linear model predictive control
        optimal_action, predicted_traj, cost = self.expert.solve(planned_traj)
        self.control_command = optimal_action

    def get_control_command(self):
        control_command_dict = self._generate_control_command()
        return control_command_dict

    def _generate_control_command(self):
        # return self.control_command
        control_command_dict = {
            "expert": self.control_command,
            "network": np.array([np.nan, np.nan, np.nan, np.nan]) if self.network_command is None else self.network_command,
            "use_network": False,
        }

        inputs = self._prepare_net_inputs()
        if not self.network_initialised:
            # apply network to init TODO: not sure what this means/why this is needed => maybe just some TF stuff?
            results = self.learner.inference(inputs)
            print("[ControllerLearning] Net initialized")
            self.network_initialised = True
            # I guess at this point the control command should be the one the expert has generated...
            # since we have to "request" that manually, will probably have to do it here...
            # TODO: why no return here?

        if self.mode != "testing" and (not self.use_network or
                                       not self.reference_updated
                                       or len(inputs["fts"].shape) != 4):
            # Will be in here if:
            # - starting and VIO init
            # - Image queue is not ready, can only run expert
            # => this should basically not happen... (in the Flightmare implementation?)
            if self.use_network:
                print("[ControllerLearning] Using expert wait for ref")
            self.n_times_expert += 1
            return control_command_dict

        # always use expert at the beginning (approximately 0.2s) to avoid synchronization problems
        # => should be a better way of doing this than this counter...
        if self.counter < 10:
            self.counter += 1
            self.n_times_expert += 1
            return control_command_dict

        # TODO: Part of the problem with having "blocking execution" and all that jazz is that the expert
        #  takes much longer to produce commands than the network. In the original DDA this is no problem since
        #  it runs completely independently in the ROS framework. However, trying to compute new expert commands
        #  at the ("targeted") network rate of 100Hz would probably take very long in practice (e.g. running sim.py
        #  with a command frequency of 100Hz). Since it doesn't really make sense to run the expert in parallel,
        #  since the simulation isn't real time and the results would therefore be skewed, one possibility would be
        #  to have two "command frequencies", one for the expert and one for the network, with the former being a lot
        #  lower (e.g. 20Hz) => this command could be manually updated at this lower frequency (e.g. the step method
        #  in PythonSimulation could return a boolean "notifying" this) and then, whenever the expert output should be
        #  used, the "cached" command is used!

        # Apply Network
        results = self.learner.inference(inputs)
        # control_command = ControlCommand()
        # control_command.armed = True
        # control_command.expected_execution_time = rospy.Time.now()
        # control_command.control_mode = 2
        # control_command.collective_thrust = results[0][0].numpy()
        # control_command.bodyrates.x = results[0][1].numpy()
        # control_command.bodyrates.y = results[0][2].numpy()
        # control_command.bodyrates.z = results[0][3].numpy()
        control_command = np.array([results[0][0], results[0][1], results[0][2], results[0][3]])
        self.network_command = control_command
        control_command_dict["network"] = self.network_command

        """
        if self.mode == "testing" and self.use_network:
            return control_command
        """

        # Log everything immediately to avoid surprises (if required)
        if self.record_data:
            self.save_data()

        # Apply random controller now and then to facilitate exploration
        if (self.mode != "testing") and random.random() < self.config.rand_controller_prob:
            self.control_command[0] += self.config.rand_thrust_mag * (random.random() - 0.5) * 2
            self.control_command[1] += self.config.rand_rate_mag * (random.random() - 0.5) * 2
            self.control_command[2] += self.config.rand_rate_mag * (random.random() - 0.5) * 2
            self.control_command[3] += self.config.rand_rate_mag * (random.random() - 0.5) * 2
            self.n_times_expert += 1
            return control_command_dict

        # Dagger (on control command label).
        d_thrust = control_command[0] - self.control_command[0]
        d_br_x = control_command[1] - self.control_command[1]
        d_br_y = control_command[2] - self.control_command[2]
        d_br_z = control_command[3] - self.control_command[3]
        # TODO: probably add if self.use_network and self.mode == "testing" or something like that
        if (self.mode == "testing" and self.use_network) \
                or (self.config.execute_nw_predictions
                    and abs(d_thrust) < self.config.fallback_threshold_rates \
                    and abs(d_br_x) < self.config.fallback_threshold_rates \
                    and abs(d_br_y) < self.config.fallback_threshold_rates \
                    and abs(d_br_z) < self.config.fallback_threshold_rates):
            self.n_times_net += 1
            control_command_dict["use_network"] = True
            return control_command_dict

        # for now just return the expert control command to see if everything works as intended/expected
        # (i.e. it should look similar to running sim.py or stuff in run_tests.py)
        self.n_times_expert += 1
        return control_command_dict

    def _prepare_net_inputs(self):
        if not self.network_initialised:
            # return fake input for init
            if self.config.use_imu:
                if self.config.use_pos:
                    n_init_states = 36
                else:
                    n_init_states = 30
            else:
                if self.config.use_pos:
                    n_init_states = 18
                else:
                    n_init_states = 15
            inputs = {"fts": np.zeros((1, self.config.seq_len, self.config.min_number_fts, 5), dtype=np.float32),
                      "state": np.zeros((1, self.config.seq_len, n_init_states), dtype=np.float32)}
            return inputs

        # reference is always used, state estimate if specified in config
        # TODO: potentially use position instead (or in addition to) body rates
        #  => probably won't require changing anything in the networks, but will for data loading
        #     and n_init_states will also have to be changed
        state_inputs = self.reference_rot + self.reference[7:].tolist()
        if self.config.use_pos:
            state_inputs += self.reference[:3].tolist()
        if self.config.use_imu:
            estimate = self.state_estimate_rot + self.state_estimate[7:].tolist()
            if self.config.use_pos:
                estimate += self.state_estimate[:3].tolist()
            state_inputs = estimate + state_inputs
        self.state_queue.append(state_inputs)

        # format the state and feature track inputs as numpy arrays for the network
        state_inputs = np.stack(self.state_queue, axis=0)
        # print("state_inputs:")
        # print(state_inputs)
        # exit()
        feature_inputs = np.stack(
            [np.stack([v for v in self.fts_queue[j].values()]) for j in range(self.config.seq_len)])
        inputs = {"fts": np.expand_dims(feature_inputs, axis=0).astype(np.float32),
                  "state": np.expand_dims(state_inputs, axis=0).astype(np.float32)}
        return inputs

    def compute_trajectory_error(self):
        gt_ref = self.reference[:3]
        gt_pos = self.state[:3]
        results = {"gt_ref": gt_ref, "gt_pos": gt_pos}
        return results

    def write_csv_header(self):
        # TODO: will need to store different stuff for our MPC => for data loading with existing framework, keep this
        row = [
            "Rollout_idx",
            "Odometry_stamp",
            # GT Position
            "gt_Position_x",
            "gt_Position_y",
            "gt_Position_z",
            "gt_Position_z_error",
            "gt_Orientation_w",
            "gt_Orientation_x",
            "gt_Orientation_y",
            "gt_Orientation_z",
            "gt_V_linear_x",
            "gt_V_linear_y",
            "gt_V_linear_z",
            "gt_V_angular_x",
            "gt_V_angular_y",
            "gt_V_angular_z",
            # VIO Estimate
            "Position_x",
            "Position_y",
            "Position_z",
            "Position_z_error",
            "Orientation_w",
            "Orientation_x",
            "Orientation_y",
            "Orientation_z",
            "V_linear_x",
            "V_linear_y",
            "V_linear_z",
            "V_angular_x",
            "V_angular_y",
            "V_angular_z",
            # Reference state
            "Reference_position_x",
            "Reference_position_y",
            "Reference_position_z",
            "Reference_orientation_w",
            "Reference_orientation_x",
            "Reference_orientation_y",
            "Reference_orientation_z",
            "Reference_v_linear_x",
            "Reference_v_linear_y",
            "Reference_v_linear_z",
            "Reference_v_angular_x",
            "Reference_v_angular_y",
            "Reference_v_angular_z",
            # MPC output with GT Position
            "Gt_control_command_collective_thrust",
            "Gt_control_command_bodyrates_x",
            "Gt_control_command_bodyrates_y",
            "Gt_control_command_bodyrates_z",
            # Net output
            "Net_control_command_collective_thrust",
            "Net_control_command_bodyrates_x",
            "Net_control_command_bodyrates_y",
            "Net_control_command_bodyrates_z",
            "Maneuver_type"
        ]

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if self.mode == "iterative":
            root_save_dir = self.config.train_dir
        else:
            root_save_dir = self.config.log_dir
        self.csv_filename = os.path.join(root_save_dir, "data_" + current_time + ".csv")
        self.image_save_dir = os.path.join(root_save_dir, "img_data_" + current_time)
        if not os.path.exists(self.image_save_dir):
            os.makedirs(self.image_save_dir)
        with open(self.csv_filename, "w") as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows([row])

    def save_data(self):
        # TODO: need to update what gets returned by the (Flightmare) simulation in terms of references and state est.
        row = [
            self.rollout_idx,
            self.simulation_time,  # time stamp
            # GT position
            self.state[0],  # pos x
            self.state[1],  # pos y
            self.state[2],  # pos z
            self.reference[2] - self.state[2],  # ref z - pos z
            self.state[3],  # rot w
            self.state[4],  # rot x
            self.state[5],  # rot y
            self.state[6],  # rot z
            self.state[7],  # vel x
            self.state[8],  # vel y
            self.state[9],  # vel z
            self.state[10],  # omega x
            self.state[11],  # omega y
            self.state[12],  # omega z
            # VIO Estimate
            self.state_estimate[0],  # pos x
            self.state_estimate[1],  # pos y
            self.state_estimate[2],  # pos z
            self.reference[2] - self.state_estimate[2],  # ref z - pos z
            self.state_estimate[3],  # rot w
            self.state_estimate[4],  # rot x
            self.state_estimate[5],  # rot y
            self.state_estimate[6],  # rot z
            self.state_estimate[7],  # vel x
            self.state_estimate[8],  # vel y
            self.state_estimate[9],  # vel z
            self.state_estimate[10],  # omega x
            self.state_estimate[11],  # omega y
            self.state_estimate[12],  # omega z
            # Reference state
            self.reference[0],  # pos x
            self.reference[1],  # pos y
            self.reference[2],  # pos z
            self.reference[3],  # rot w
            self.reference[4],  # rot x
            self.reference[5],  # rot y
            self.reference[6],  # rot z
            self.reference[7],  # vel x
            self.reference[8],  # vel y
            self.reference[9],  # vel z
            self.reference[10],  # omega x
            self.reference[11],  # omega y
            self.reference[12],  # omega z
            # MPC output with GT Position
            self.control_command[0],  # collective thrust
            self.control_command[1],  # roll
            self.control_command[2],  # pitch
            self.control_command[3],  # yaw
            # NET output with GT Position
            self.network_command[0],  # collective thrust
            self.network_command[1],  # roll
            self.network_command[2],  # pitch
            self.network_command[3],  # yaw
            # Maneuver type
            0,
        ]

        # if self.record_data and self.gt_odometry.pose.pose.position.z > 0.3 and \
        #         self.control_command.collective_thrust > 0.2:  TODO: not sure if these are really needed
        if self.record_data:
            with open(self.csv_filename, "a") as writeFile:
                writer = csv.writer(writeFile)
                writer.writerows([row])
            fts_name = "{:08d}.npy"
            fts_filename = os.path.join(self.image_save_dir, fts_name.format(self.recorded_samples))
            np.save(fts_filename, self.feature_tracks)
            self.recorded_samples += 1
