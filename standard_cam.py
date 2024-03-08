import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, Input, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model

class ChannelAttention(layers.Layer):
    def __init__(self, channels, reduction_rate=16, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.avg_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.max_pool = layers.GlobalMaxPooling2D(keepdims=True)
        self.excitation = models.Sequential([
            layers.Conv2D(filters=channels // reduction_rate, kernel_size=1),
            layers.ReLU(),
            layers.Conv2D(filters=channels, kernel_size=1),
        ])
        self.sigmoid = layers.Activation('sigmoid')

    def call(self, inputs):
        avg_pool = self.avg_pool(inputs)
        max_pool = self.max_pool(inputs)
        avg_out = self.excitation(avg_pool)
        max_out = self.excitation(max_pool)
        attention = self.sigmoid(avg_out + max_out)
        return attention * inputs

class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.conv = layers.Conv2D(filters=1, kernel_size=kernel_size, padding='same')
        self.sigmoid = layers.Activation('sigmoid')

    def call(self, inputs):
        avg_feat = tf.reduce_mean(inputs, axis=3, keepdims=True)
        max_feat = tf.reduce_max(inputs, axis=3, keepdims=True)
        feat = tf.concat([avg_feat, max_feat], axis=3)
        out_feat = self.conv(feat)
        attention = self.sigmoid(out_feat)
        return attention * inputs

class CBAM(layers.Layer):
    def __init__(self, channels, reduction_rate=4, kernel_size=7, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.channel_attention = ChannelAttention(channels, reduction_rate)
        self.spatial_attention = SpatialAttention(kernel_size)

    def call(self, inputs):
        out = self.channel_attention(inputs)
        out = self.spatial_attention(out)
        return out

class LargeKernelAttnLayer(layers.Layer):
    def __init__(self, channels, **kwargs):
        super(LargeKernelAttnLayer, self).__init__(**kwargs)
        self.channels = channels
        self.dwconv = layers.DepthwiseConv2D(kernel_size=5, padding='same', depth_multiplier=1)
        self.dwdconv = layers.DepthwiseConv2D(kernel_size=7, padding='same', depth_multiplier=1, dilation_rate=3)
        self.pwconv = layers.Conv2D(filters=self.channels, kernel_size=1)

    def call(self, inputs):
        weight = self.pwconv(self.dwdconv(self.dwconv(inputs)))
        return inputs * weight

    def get_config(self):
        config = super(LargeKernelAttnLayer, self).get_config()
        config.update({"channels": self.channels})
        return config
