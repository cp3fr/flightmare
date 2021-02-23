# TODO: implement feature extraction from images to attention in some form
#  e.g. simply encoder features, "attention tracks" etc. => should probably have "general interface"/superclass
import torch
import torch.nn.functional as F
import cv2
import numpy as np

from torchvision import transforms
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
        original_image = image.copy()
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

        # get the gaze prediction output
        gaze_location = out["output_gaze"].squeeze().cpu().detach().numpy()

        # transform into normalised image coordinates (also, to be consistent with the other gaze tracks,
        # x-coords from left to right, y-coords from top to bottom => need to be inverted)
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
