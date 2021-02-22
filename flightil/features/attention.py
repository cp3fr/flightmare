# TODO: implement feature extraction from images to attention in some form
#  e.g. simply encoder features, "attention tracks" etc. => should probably have "general interface"/superclass
import torch
import torch.nn.functional as F
import cv2

from torchvision import transforms
from gazesim.models.utils import image_softmax, convert_attention_to_image
from gazesim.training.utils import load_model, to_batch, to_device


class AttentionFeatures:

    def __init__(self):
        self.model = None

    def get_attention_features(self, image, **kwargs):
        raise NotImplementedError()


class AttentionDecoderFeatures(AttentionFeatures):

    def __init__(self, config):
        super().__init__()

        print("\n[AttentionDecoderFeatures] Loading attention model.\n")

        # load model and move to correct device
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:{}".format(config.gpu) if use_cuda else "cpu")
        self.model, self.model_config = load_model(config.attention_model_path, gpu=config.gpu, return_config=True)
        self.model.to(self.device)

        # set model to only return features
        self.model.features_only = True

        # prepare transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.model_config["resize"]),
            transforms.ToTensor(),
            # transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
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
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:{}".format(config.gpu) if use_cuda else "cpu")
        self.model, self.model_config = load_model(config.attention_model_path, gpu=config.gpu, return_config=True)
        self.model.to(self.device)

        # prepare transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.model_config["resize"]),
            transforms.ToTensor(),
        ])

        # keeping track of the time to calculate "attention velocity"
        self.previous_time = -1

        print("\n[AttentionMapTracks] Done loading attention model.\n")

    def get_attention_features(self, image, **kwargs):
        print(image.shape)

        # convert from OpenCV image format (BGR) to normal/PIMS format (RGB) which was used for training
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # apply transform
        image = self.transform(image)

        # prepare batch for input to the model
        batch = {"input_image_0": image}
        batch = to_device(to_batch([batch]), self.device)

        # get the network output
        out = self.model(batch)

        print(image.shape)
        print(out["output_attention"].shape)

        # get the attention map output and "activate" it
        attention_map = convert_attention_to_image(image_softmax((out["output_attention"])),
                                                   out_shape=image.shape[:2]).cpu().detach().numpy()
        print(attention_map.shape)
        # TODO: need to also check whether input dimensions are correct in this order...
        # TODO: need to check what the shape/min/max of this is

        # TODO: get the maximum or something like that as an "attention track" (including velocity)
        current_time = kwargs.get("current_time", -1)
        time_diff = current_time - self.previous_time


class GazeTracks(AttentionFeatures):

    # TODO: maybe different kinds of tracks, either from the attention map predictions or regressed positions

    def __init__(self):
        super().__init__()

    def get_attention_features(self, image, **kwargs):
        pass
