class ChannelAttention(layers.Layer):
    def __init__(self, channels, ratio=8):
        super(ChannelAttention, self).__init__()
        self.channels = channels
        self.ratio = ratio

    def build(self, channels):
        self.gap_dense = Conv2D(self.channels // self.ratio, kernel_size=1, use_bias=True, kernel_regularizer=l2(0.001), kernel_initializer='he_normal')
        self.gmp_dense = Conv2D(self.channels // self.ratio, kernel_size=1, use_bias=True, kernel_regularizer=l2(0.001), kernel_initializer='he_normal')
        self.shared_dense_one = Conv2D(self.channels, kernel_size=1, use_bias=True, kernel_regularizer=l2(0.001), kernel_initializer='he_normal')
        self.shared_bn = BatchNormalization()
        super(ChannelAttention, self).build(channels)

    def call(self, inputs):
        gapavg = GlobalAveragePooling2D()(inputs)
        gapavg = Reshape((1, 1, gapavg.shape[-1]))(gapavg)
        gapavg = self.gap_dense(gapavg)
        gapavg = self.shared_bn(gapavg)
        gapavg = Activation('relu')(gapavg)
        gapavg_out = self.shared_dense_one(gapavg)

        gmpmax = GlobalMaxPooling2D()(inputs)
        gmpmax = Reshape((1, 1, gmpmax.shape[-1]))(gmpmax)
        gmpmax = self.gmp_dense(gmpmax)
        gmpmax = self.shared_bn(gmpmax)
        gmpmax = Activation('relu')(gmpmax)
        gmpmax_out = self.shared_dense_one(gmpmax)

        channel_attention = tf.sigmoid(add([gapavg_out, gmpmax_out]))
        return inputs * channel_attention

class SpatialAttention(layers.Layer):
    def __init__(self):
        super(SpatialAttention, self).__init__()

    def build(self, channels):
        self.conv1 = Conv2D(filters=1, kernel_size=7, strides=1, padding='same', use_bias=False, kernel_regularizer=l2(0.001), kernel_initializer='he_normal')
        self.bn = BatchNormalization()
        super(SpatialAttention, self).build(channels)

    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_out = tf.reduce_max(inputs, axis=-1, keepdims=True)
        x = tf.concat([avg_out, max_out], axis=-1)
        x = self.conv1(x)
        x = self.bn(x)
        spatial_attention = tf.sigmoid(x)
        return inputs * spatial_attention

class CBAM(layers.Layer):
    def __init__(self, channels, ratio=8):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, ratio)
        self.spatial_attention = SpatialAttention()

    def call(self, inputs):
        x = self.channel_attention(inputs)
        x = self.spatial_attention(x)
        return x
