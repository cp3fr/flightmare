import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, LeakyReLU, Conv1D
from tensorflow.keras.layers import Flatten, GlobalAveragePooling2D
from tensorflow.keras.activations import sigmoid, tanh

try:
    from .tf_addons_normalizations import InstanceNormalization
except:
    from tf_addons_normalizations import InstanceNormalization


def create_network(settings):
    net = AggressiveNet(settings)
    return net


class Network(Model):
    def __init__(self):
        super(Network, self).__init__()

    def create(self):
        self._create()

    def call(self, x, **kwargs):
        return self._internal_call(x)

    def _create(self, *args, **kwargs):
        raise NotImplementedError

    def _internal_call(self, inputs):
        raise NotImplementedError


class AggressiveNet(Network):

    def __init__(self, config):
        super(AggressiveNet, self).__init__()
        self.config = config
        self.multiplier = tf.constant([21.0, 6.0, 6.0, 6.0])
        self._create(input_size=(config.seq_len, config.min_number_fts, 5))

    def _create(self, input_size, has_bias=True, learn_affine=True):
        """Init.
        Args:
            input_size (tuple): size of input
            has_bias (bool, optional): Defaults to True. Conv1d bias?
            learn_affine (bool, optional): Defaults to True. InstanceNorm1d affine?
        """
        if self.config.use_fts_tracks:
            f = 2.0
            self.pointnet = [
                Conv2D(int(16 * f), kernel_size=1, strides=1, padding="valid", dilation_rate=1,
                       use_bias=has_bias, input_shape=input_size),
                InstanceNormalization(axis=3, epsilon=1e-5, center=learn_affine, scale=learn_affine),
                LeakyReLU(alpha=1e-2),
                Conv2D(int(32 * f), kernel_size=1, strides=1, padding="valid", dilation_rate=1, use_bias=has_bias),
                InstanceNormalization(axis=3, epsilon=1e-5, center=learn_affine, scale=learn_affine),
                LeakyReLU(alpha=1e-2),
                Conv2D(int(64 * f), kernel_size=1, strides=1, padding="valid", dilation_rate=1, use_bias=has_bias),
                InstanceNormalization(axis=3, epsilon=1e-5, center=learn_affine, scale=learn_affine),
                LeakyReLU(alpha=1e-2),
                Conv2D(int(64 * f), kernel_size=1, strides=1, padding="valid", dilation_rate=1, use_bias=has_bias),
                GlobalAveragePooling2D()
            ]

            input_size = (self.config.seq_len, int(64 * f))
            self.fts_mergenet = [
                Conv1D(int(64 * f), kernel_size=2, strides=1, padding="same", dilation_rate=1, input_shape=input_size),
                LeakyReLU(alpha=1e-2),
                Conv1D(int(32 * f), kernel_size=2, strides=1, padding="same", dilation_rate=1),
                LeakyReLU(alpha=1e-2),
                Conv1D(int(32 * f), kernel_size=2, strides=1, padding="same", dilation_rate=1),
                LeakyReLU(alpha=1e-2),
                Conv1D(int(32 * f), kernel_size=2, strides=1, padding="same", dilation_rate=1),
                LeakyReLU(alpha=1e-2),
                Flatten(),
                Dense(int(64 * f))
            ]

        g = 2.0
        if self.config.attention_fts_type != "none":
            # TODO: do we not need input_shape here? also, does it matter which features are selected
            #  I think it shouldn't matter since they basically only differ in the channel dimension, which
            #  TensorFlow seems to calculate on the fly
            self.attention_fts_conv = [
                Conv1D(int(64 * g), kernel_size=2, strides=1, padding="same", dilation_rate=1),
                LeakyReLU(alpha=1e-2),
                Conv1D(int(32 * g), kernel_size=2, strides=1, padding="same", dilation_rate=1),
                LeakyReLU(alpha=1e-2),
                Conv1D(int(32 * g), kernel_size=2, strides=1, padding="same", dilation_rate=1),
                LeakyReLU(alpha=1e-2),
                Conv1D(int(32 * g), kernel_size=2, strides=1, padding="same", dilation_rate=1),
                Flatten(),
                Dense(int(64 * g))
            ]

        self.states_conv = [
            Conv1D(int(64 * g), kernel_size=2, strides=1, padding="same", dilation_rate=1),
            LeakyReLU(alpha=1e-2),
            Conv1D(int(32 * g), kernel_size=2, strides=1, padding="same", dilation_rate=1),
            LeakyReLU(alpha=1e-2),
            Conv1D(int(32 * g), kernel_size=2, strides=1, padding="same", dilation_rate=1),
            LeakyReLU(alpha=1e-2),
            Conv1D(int(32 * g), kernel_size=2, strides=1, padding="same", dilation_rate=1),
            Flatten(),
            Dense(int(64 * g))
        ]

        # TODO: makes these lists of lists for "attention-induced" branching
        if self.config.attention_fts_type == "branching":
            pass
        if self.config.shallow_control_module:
            self.control_module = [
                Dense(4)
            ]
        else:
            self.control_module = [
                Dense(64 * g),
                LeakyReLU(alpha=1e-2),
                Dense(32 * g),
                LeakyReLU(alpha=1e-2),
                Dense(16 * g),
                LeakyReLU(alpha=1e-2),
                Dense(4)
            ]

    def _pointnet_branch(self, single_t_features):
        x = tf.expand_dims(single_t_features, axis=1)
        for f in self.pointnet:
            x = f(x)
        return x

    def _features_branch(self, input_features):
        preprocessed_fts = tf.map_fn(self._pointnet_branch,
                                     elems=input_features,
                                     parallel_iterations=self.config.seq_len)  # (seq_len, batch_size, 64)
        preprocessed_fts = tf.transpose(preprocessed_fts, (1, 0, 2))  # (batch_size, seq_len, 64)
        x = preprocessed_fts
        for f in self.fts_mergenet:
            x = f(x)
        return x

    def _attention_fts_branch(self, features):
        x = features
        for f in self.attention_fts_conv:
            x = f(x)
        return x

    def _states_branch(self, embeddings):
        x = embeddings
        for f in self.states_conv:
            x = f(x)
        return x

    def _control_branch(self, embeddings, branch=None):
        x = embeddings
        for f in self.control_module:
            x = f(x)
        return x

    def _internal_call(self, inputs):
        states = inputs["state"]

        # get the feature track embeddings if specified
        fts_embeddings = None
        if self.config.use_fts_tracks:
            fts_stack = inputs["fts"]  # (batch_size, seq_len, min_numb_features, 5)
            fts_stack = tf.transpose(fts_stack, (1, 0, 2, 3))  # (seq_len, batch_size, min_numb_features, 5)
            # Execute PointNet Part
            fts_embeddings = self._features_branch(fts_stack)

        # get the attention feature embeddings if specified
        attention_fts_embeddings = None
        if self.config.attention_fts_type != "none":
            attention_fts = inputs["attention_fts"]
            attention_fts_embeddings = self._attention_fts_branch(attention_fts)

        # get the state (reference and state estimate) embeddings
        states_embeddings = self._states_branch(states)

        # concatenate the embeddings
        if self.config.use_fts_tracks:
            total_embeddings = tf.concat((fts_embeddings, states_embeddings), axis=1)
        else:
            total_embeddings = states_embeddings

        if self.config.attention_fts_type != "none":
            total_embeddings = tf.concat((total_embeddings, attention_fts_embeddings), axis=1)

        # get the output of the final "control" module
        # TODO: probably add branching input
        output = self._control_branch(total_embeddings, branch=inputs.get("branch", None))

        # apply different activation functions to the different components
        if self.config.use_activation:
            thrust_tensor, body_rate_tensor = tf.split(output, [1, 3], axis=-1)
            thrust_tensor = sigmoid(thrust_tensor)
            body_rate_tensor = tanh(body_rate_tensor)
            output = tf.concat((thrust_tensor, body_rate_tensor), axis=-1)
            output = tf.multiply(output, self.multiplier)

        return output
