# TODO: implement feature extraction from images to attention in some form
#  e.g. simply encoder features, "attention tracks" etc. => should probably have "general interface"/superclass
import torch
import torch.nn.functional as F
import cv2
import numpy as np

from torchvision import transforms
from scipy.spatial.transform import Rotation
from gazesim.models.utils import image_softmax, convert_attention_to_image
from gazesim.training.utils import load_model, to_batch, to_device


class AttentionFeatures:

    def get_attention_features(self, image, **kwargs):
        raise NotImplementedError()


class AttentionDecoderFeatures(AttentionFeatures):

    def __init__(self, config):
        super().__init__()

        print("\n[AttentionDecoderFeatures] Loading attention model.\n")

        # load model and move to correct device
        load_path = config.attention_model_path if isinstance(config.attention_model_path, str) else config.attention_model_path[0]
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:{}".format(config.gpu) if use_cuda else "cpu")
        self.model, self.model_config = load_model(load_path, gpu=config.gpu, return_config=True)
        self.model.to(self.device)

        # set model to only return features
        self.model.features_only = True

        # prepare transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.model_config["resize"]),
            transforms.ToTensor(),
        ])

        print("\n[AttentionDecoderFeatures] Done loading attention model.\n")

    def get_attention_features(self, image, **kwargs):
        # convert from OpenCV image format (BGR) to normal/PIMS format (RGB) which was used for training
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # apply transform
        image = self.transform(image)

        # prepare batch for input to the model
        batch = {"input_image_0": image}
        batch = to_device(to_batch([batch]), self.device)

        # get the network output
        out = self.model(batch)

        # get the features .reshape(-1).cpu().detach().numpy()
        attention_features = F.adaptive_avg_pool2d(out["output_features"], (1, 1)).reshape(-1).cpu().detach().numpy()

        return attention_features


class AttentionMapTracks(AttentionFeatures):

    def __init__(self, config):
        super().__init__()

        print("\n[AttentionMapTracks] Loading attention model.\n")

        # load model and move to correct device
        load_path = config.attention_model_path if isinstance(config.attention_model_path, str) else config.attention_model_path[0]
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:{}".format(config.gpu) if use_cuda else "cpu")
        self.model, self.model_config = load_model(load_path, gpu=config.gpu, return_config=True)
        self.model.to(self.device)

        # prepare transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.model_config["resize"]),
            transforms.ToTensor(),
        ])

        # keeping track of the time and previous point to calculate "attention velocity"
        self.previous_time = -1
        self.previous_gaze_location = None

        print("\n[AttentionMapTracks] Done loading attention model.\n")

    def get_attention_features(self, image, **kwargs):
        current_time = kwargs.get("current_time", -1)

        # convert from OpenCV image format (BGR) to normal/PIMS format (RGB) which was used for training
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # apply transform
        image = self.transform(image)

        # prepare batch for input to the model
        batch = {"input_image_0": image}
        batch = to_device(to_batch([batch]), self.device)

        # get the network output
        out = self.model(batch)

        # get the attention map output and "activate" it
        attention_map = convert_attention_to_image(
            attention=image_softmax((out["output_attention"])),
            out_shape=out["output_attention"].shape[2:],
        ).squeeze().cpu().detach().numpy()

        # these are "OpenCV coordinates", i.e. from the top left to the bottom right
        grid_indices = np.mgrid[0:attention_map.shape[0], 0:attention_map.shape[1]].transpose(1, 2, 0).reshape(-1, 2)
        gaze_location = np.average(grid_indices, axis=0, weights=attention_map.reshape(-1))
        gaze_location = (gaze_location / np.array([attention_map.shape[0], attention_map.shape[1]]) * 2.0) - 1.0

        # calculate the velocity if possible
        if self.previous_gaze_location is not None and current_time >= 0 and self.previous_time >= 0:
            gaze_velocity = (gaze_location - self.previous_gaze_location) - (current_time - self.previous_time)
        else:
            gaze_velocity = np.zeros((2,), dtype=gaze_location.dtype)
        self.previous_time = current_time
        self.previous_gaze_location = gaze_location

        attention_track = np.array([gaze_location[0], gaze_location[1], gaze_velocity[0], gaze_velocity[1]])
        return attention_track


class GazeTracks(AttentionFeatures):

    def __init__(self, config):
        super().__init__()

        print("\n[GazeTracks] Loading attention model.\n")

        # load model and move to correct device
        load_path = config.attention_model_path if isinstance(config.attention_model_path, str) else config.attention_model_path[1]
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:{}".format(config.gpu) if use_cuda else "cpu")
        self.model, self.model_config = load_model(load_path, gpu=config.gpu, return_config=True)
        self.model.to(self.device)

        # prepare transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.model_config["resize"]),
            transforms.ToTensor(),
        ])

        # keeping track of the time and previous point to calculate "attention velocity"
        self.previous_time = -1
        self.previous_gaze_location = None

        # TODO: clean this up
        self.scale_gaze = True

        print("\n[GazeTracks] Done loading attention model.\n")

    def get_attention_features(self, image, **kwargs):
        current_time = kwargs.get("current_time", -1)

        # convert from OpenCV image format (BGR) to normal/PIMS format (RGB) which was used for training
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # apply transform
        image_tf = self.transform(image_rgb)

        # prepare batch for input to the model
        batch = {"input_image_0": image_tf}
        batch = to_device(to_batch([batch]), self.device)

        # get the network output
        out = self.model(batch)

        # get the gaze prediction output
        gaze_location = out["output_gaze"].squeeze().cpu().detach().numpy()

        # transform into normalised image coordinates (also, to be consistent with the other gaze tracks,
        # x-coords from left to right, y-coords from top to bottom)
        if self.scale_gaze:
            gaze_location = (gaze_location / 2.0) / np.array([400.0, 300.0])
        gaze_location = gaze_location[::-1]

        # calculate the velocity if possible
        if self.previous_gaze_location is not None and current_time >= 0 and self.previous_time >= 0:
            gaze_velocity = (gaze_location - self.previous_gaze_location) - (current_time - self.previous_time)
        else:
            gaze_velocity = np.zeros((2,), dtype=gaze_location.dtype)
        self.previous_time = current_time
        self.previous_gaze_location = gaze_location

        attention_track = np.array([gaze_location[0], gaze_location[1], gaze_velocity[0], gaze_velocity[1]])
        return attention_track


class AttentionHighLevelLabel(AttentionFeatures):

    def __init__(self, config):
        # TODO: probably use one of the two gaze models to do stuff...
        #  => would be good to do offline evaluation of both of them in some way
        #     to e.g. see whether the map to gaze model actually works well/better

        self.camera_fov = 80
        self.camera_uptilt_angle = -(30.0 / 90.0) * (np.pi / 2)
        self.camera_pos_body_frame = np.array([0.2, 0.0, 0.1])
        self.camera_rot_body_frame = Rotation.from_quat([
            0.0, np.sin(0.5 * self.camera_uptilt_angle),
            0.0, np.cos(0.5 * self.camera_uptilt_angle)
        ])
        self.camera_matrix = np.array([
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.0, 0.0, 1.0],
        ])

        self.gaze_extractor = GazeTracks(config)
        # self.gaze_extractor = AttentionMapTracks(config)

        self.decision_threshold = config.attention_branching_threshold

    def get_attention_features(self, image, **kwargs):
        """
        test = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        test = cv2.circle(test, (400, 300), 5, (0, 0, 255), -1)
        cv2.imshow("", test)
        cv2.waitKey(0)
        """

        drone_state = kwargs.get("drone_state", np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        drone_pos = drone_state[:3]
        drone_rot = Rotation.from_quat(drone_state[4:7].tolist() + drone_state[3:4].tolist())
        drone_vel = drone_state[7:10]
        # drone_vel = np.array([-1.0, -0.10, 0.0])

        # transform camera pos/rot from drone body frame to world frame
        camera_pos_world_frame = drone_rot.apply(self.camera_pos_body_frame) + drone_pos
        camera_rot_world_frame = (drone_rot * self.camera_rot_body_frame)

        print("\nDrone position:", drone_pos)
        print("Camera position:", camera_pos_world_frame)
        print("Test:", drone_rot.apply([1.0, 0.0, 0.3]) + drone_pos)

        print("Drone rotation (Euler angles):", drone_rot.as_euler("xyz", degrees=True))
        print("Camera rotation (Euler angles):", camera_rot_world_frame.as_euler("xyz", degrees=True))

        # get the gaze vector in 2D in the correct format (range [0, 1], x=right, y=down)
        gaze_2d = self.gaze_extractor.get_attention_features(image, **kwargs)
        print("Predicted gaze 2D (1):", gaze_2d)
        gaze_2d = gaze_2d[:2][::-1]  # need only position, but it's ordered in "OpenCV indexing" => y first, x second
        print("Predicted gaze 2D (2):", gaze_2d)
        gaze_2d = (gaze_2d + 1.0) / 2.0
        print("Predicted gaze 2D (3):", gaze_2d)
        # gaze_2d = np.array([0.5, 1.0])

        cv2_gaze_2d = tuple((gaze_2d * np.array([800, 600])).astype(int))

        # get the gaze vector in 3D
        gaze_2d = np.hstack((gaze_2d, [1.0]))
        gaze_3d = (np.linalg.pinv(self.camera_matrix @ np.eye(3)) @ gaze_2d.T).T
        gaze_3d = np.array([gaze_3d[2], -gaze_3d[0], -gaze_3d[1]])
        gaze_3d = gaze_3d / np.linalg.norm(gaze_3d)
        gaze_3d = camera_rot_world_frame.apply(gaze_3d)

        print("Velocity vector 3D:", drone_vel / np.linalg.norm(drone_vel))
        print("Gaze 2D:", gaze_2d)
        print("Gaze 3D:", gaze_3d)

        # compute the signed angle between the gaze and the velocity vector
        drone_vel_2d = drone_vel[[0, 1]]
        drone_vel_2d = drone_vel_2d / np.linalg.norm(drone_vel_2d)
        gaze_2d = gaze_3d[[0, 1]]
        gaze_2d = gaze_2d / np.linalg.norm(gaze_2d)
        angle = np.arctan2(drone_vel_2d[1], drone_vel_2d[0]) - np.arctan2(gaze_2d[1], gaze_2d[0])
        print("Angle (1): {} (rad), {} (deg)".format(angle, angle * 180.0 / np.pi))
        if np.abs(angle) > np.pi:
            if angle < 0.0:
                # angle = 2 * np.pi - np.abs(angle)
                angle = angle + 2 * np.pi
                print("Angle (2): {} (rad), {} (deg)".format(angle, angle * 180.0 / np.pi))
            else:
                # angle = -(2 * np.pi - np.abs(angle))
                angle = angle - 2 * np.pi
                print("Angle (3): {} (rad), {} (deg)".format(angle, angle * 180.0 / np.pi))

            """
            if np.abs(angle) > 0.0:
                angle = -angle
                print("Angle (4): {} (rad), {} (deg)".format(angle, angle * 180.0 / np.pi))
            """

        print("Angle: {} (rad), {} (deg)".format(angle, angle * 180.0 / np.pi))
        angle = angle * 180.0 / np.pi

        high_level_label = 0
        if angle > self.decision_threshold:
            high_level_label = 1
        elif angle < -self.decision_threshold:
            high_level_label = 2

        print("High-level label:", high_level_label, "\n")

        # img_to_show = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cv2.COLOR_RGB2BGR)
        # img_to_show = cv2.circle(img_to_show, cv2_gaze_2d, 5, (0, 0, 255, -1))
        # cv2.imshow("", img_to_show)
        # cv2.waitKey(0)

        return high_level_label


class AllAttentionFeatures(AttentionFeatures):

    def __init__(self, config):
        self.attention_decoder_features = AttentionDecoderFeatures(config)
        self.attention_map_tracks = AttentionMapTracks(config)
        self.gaze_tracks = GazeTracks(config)

    def get_attention_features(self, image, **kwargs):
        out = {
            "decoder_fts": self.attention_decoder_features.get_attention_features(image, **kwargs),
            "map_tracks": self.attention_map_tracks.get_attention_features(image, **kwargs),
            "gaze_tracks": self.gaze_tracks.get_attention_features(image, **kwargs),
        }
        return out


if __name__ == "__main__":
    extractor = AttentionHighLevelLabel()
    test = extractor.get_attention_features(None)
