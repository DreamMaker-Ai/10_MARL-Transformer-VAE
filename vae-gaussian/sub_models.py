import os.path

import tensorflow as tf
import numpy as np

from config import Config
from utils_transformer import make_mask, make_padded_obs


class CNNModel(tf.keras.models.Model):
    """
    # Add AutoEncoder
    :param obs_shape: (15,15,6)
    :param max_num_agents=n=15
    :return: (None,n,hidden_dim),(None,n,hidden_dim),(None,n,hidden_dim),(None,n,15,15,6),
                =(None,15,256),(None,15,256),(None,15,256),(None,15,15,15,256)

    Model: "cnn_model"
    _______________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to
    ===============================================================================================
     input_1 (InputLayer)           [(None, 15, 15, 15,  0           []
                                     6)]

     time_distributed (TimeDistribu  (None, 15, 15, 15,   448        ['input_1[0][0]']
     ted)                           64)

     time_distributed_1 (TimeDistri  (None, 15, 7, 7, 64  36928      ['time_distributed[0][0]']
     buted)                         )

     time_distributed_2 (TimeDistri  (None, 15, 5, 5, 64  36928      ['time_distributed_1[0][0]']
     buted)                         )

     time_distributed_3 (TimeDistri  (None, 15, 3, 3, 64  36928      ['time_distributed_2[0][0]']
     buted)                         )

     time_distributed_4 (TimeDistri  (None, 15, 576)     0           ['time_distributed_3[0][0]']
     buted)

     time_distributed_5 (TimeDistri  (None, 15, 256)     147712      ['time_distributed_4[0][0]']
     buted)

     time_distributed_6 (TimeDistri  (None, 15, 256)     65792       ['time_distributed_5[0][0]']
     buted)

     time_distributed_7 (TimeDistri  (None, 15, 256)     65792       ['time_distributed_5[0][0]']
     buted)

     lambda (Lambda)                (None, 15, 256)      0           ['time_distributed_6[0][0]',
                                                                      'time_distributed_7[0][0]']

     tf.math.multiply_2 (TFOpLambda  (None, 15, 256)     0           ['lambda[0][0]']
     )

     time_distributed_8 (TimeDistri  (None, 15, 576)     148032      ['tf.math.multiply_2[0][0]']
     buted)

     time_distributed_9 (TimeDistri  (None, 15, 3, 3, 64  0          ['time_distributed_8[0][0]']
     buted)                         )

     time_distributed_10 (TimeDistr  (None, 15, 5, 5, 64  36928      ['time_distributed_9[0][0]']
     ibuted)                        )

     time_distributed_11 (TimeDistr  (None, 15, 7, 7, 64  36928      ['time_distributed_10[0][0]']
     ibuted)                        )

     time_distributed_12 (TimeDistr  (None, 15, 15, 15,   36928      ['time_distributed_11[0][0]']
     ibuted)                        64)

     time_distributed_13 (TimeDistr  (None, 15, 15, 15,   390        ['time_distributed_12[0][0]']
     ibuted)                        6)

     tf.math.multiply (TFOpLambda)  (None, 15, 256)      0           ['time_distributed_6[0][0]']

     tf.math.multiply_1 (TFOpLambda  (None, 15, 256)     0           ['time_distributed_7[0][0]']
     )

     tf.math.multiply_3 (TFOpLambda  (None, 15, 15, 15,   0          ['time_distributed_13[0][0]']
     )                              6)

    ===============================================================================================
    Total params: 649,734
    Trainable params: 649,734
    Non-trainable params: 0
    _______________________________________________________________________________________________
    """

    def __init__(self, config, **kwargs):
        super(CNNModel, self).__init__(**kwargs)

        self.config = config

        self.conv0 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=1,
                    strides=1,
                    activation='relu',
                    kernel_initializer='Orthogonal'
                )
            )

        self.conv1 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=3,
                    strides=2,
                    activation='relu',
                    kernel_initializer='Orthogonal'
                )
            )

        self.conv2 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=3,
                    strides=1,
                    activation='relu',
                    kernel_initializer='Orthogonal'
                )
            )

        self.conv3 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=3,
                    strides=1,
                    activation='relu',
                    kernel_initializer='Orthogonal'
                )
            )

        self.flatten1 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Flatten()
            )

        self.z_mu = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=self.config.latent_dim,
                    activation=None,
                )
            )

        self.z_logvar = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=self.config.latent_dim,
                    activation=None,
                )
            )

        self.z = \
            tf.keras.layers.Lambda(
                self.sampling,
                output_shape=(self.config.max_num_red_agents,
                              self.config.latent_dim),
            )

        self.dense2 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=576,
                    activation='relu',
                )
            )

        self.reshape = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Reshape(target_shape=(3, 3, 64))
            )

        self.deconv1 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=3,
                    strides=1,
                    activation='relu',
                )
            )

        self.deconv2 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=3,
                    strides=1,
                    activation='relu',
                )
            )

        self.deconv3 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2DTranspose(
                    filters=64,
                    kernel_size=3,
                    strides=2,
                    activation='relu',
                )
            )

        self.deconv4 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2DTranspose(
                    filters=6,
                    kernel_size=1,
                    strides=1,
                    activation='sigmoid',
                )
            )

    def sampling(self, args):
        mu, log_var = args  # (1,15,32)
        batch = tf.keras.backend.shape(mu)[0]
        seq = tf.keras.backend.int_shape(mu)[1]
        dim = tf.keras.backend.int_shape(mu)[2]

        epsilon = tf.random.normal(shape=(batch, seq, dim))  # (1,15,256)

        return mu + tf.exp(log_var / 2) * epsilon  # (1,15,256)

    @tf.function
    def encoder(self, inputs):
        # inputs: (b,n,g,g,ch*n_frames)=(1,15,15,15,6)

        h = self.conv0(inputs)  # (1,15,15,15,64)
        h = self.conv1(h)  # (1,15,7,7,64)
        h = self.conv2(h)  # (1,15,5,5,64)
        h = self.conv3(h)  # (1,15,3,3,64)

        h1 = self.flatten1(h)  # (1,17,576)

        z_mu = self.z_mu(h1)  # (1,15,256)
        z_logvar = self.z_logvar(h1)  # (1,15,256)

        z = self.z([z_mu, z_logvar])  # (1,15,256)

        return z_mu, z_logvar, z  # (1,15,256), (1,15,256), (1,15,256)

    @tf.function
    def decoder(self, features):
        d1 = self.dense2(features)  # (1,15,256)

        d1 = self.reshape(d1)  # (1,15,3,3,64)

        d = self.deconv1(d1)  # (1,15,5,5,64)
        d = self.deconv2(d)  # (1,15,7,7,64)
        d = self.deconv3(d)  # (1,15,15,15,64)
        reconst = self.deconv4(d)  # (1,15,15,15,6)

        return reconst

    @tf.function
    def call(self, inputs, mask):
        # inputs: (b,n,g,g,ch*n_frames)=(1,15,15,15,6)
        # mask: (b,n)=(1,15), bool

        # encoder
        z_mu, z_logvar, z = self.encoder(inputs)

        broadcast_float_mask = \
            tf.expand_dims(
                tf.cast(mask, 'float32'),
                axis=-1
            )  # Add feature dim for broadcast, (1,15,1)

        z_mu = z_mu * broadcast_float_mask  # (1,15,256)
        z_logvar = z_logvar * broadcast_float_mask  # (1,15,256)

        features = z * broadcast_float_mask  # (1,15,256)

        # decoder
        reconst = self.decoder(features)  # (1,15,15,15,6)

        broadcast_float_mask = \
            tf.expand_dims(
                tf.cast(broadcast_float_mask, 'float32'),
                axis=-1
            )  # Add feature dim for broadcast, (1,15,1,1)

        broadcast_float_mask = \
            tf.expand_dims(
                tf.cast(broadcast_float_mask, 'float32'),
                axis=-1
            )  # Add feature dim for broadcast, (1,15,1,1,1)

        reconst = reconst * broadcast_float_mask  # (1,15,15,15,6)

        return z_mu, z_logvar, features, reconst

    def build_graph(self, mask):
        """ For summary & plot_model """
        x = tf.keras.layers.Input(
            shape=(self.config.max_num_red_agents,
                   self.config.grid_size,
                   self.config.grid_size,
                   self.config.observation_channels * self.config.n_frames)
        )

        model = \
            tf.keras.models.Model(
                inputs=[x],
                outputs=self.call(x, mask),
                name='cnn_model'
            )

        return model


class MultiHeadAttentionModel(tf.keras.models.Model):
    """
    Two layers of MultiHeadAttention (Self Attention with provided mask)

    :param mask: (None,n,n), bool
    :param max_num_agents=15=n
    :param hidden_dim = 256

    :return: features: (None,n,hidden_dim)=(None,15,256)
             score: (None,num_heads,n,n)=(None,2,15,15)

    Model: "mha_1"
    _______________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to
    ===============================================================================================
     input_3 (InputLayer)           [(None, 15, 256)]    0           []

     multi_head_attention (MultiHea  ((None, 15, 256),   263168      ['input_3[0][0]',
     dAttention)                     (None, 2, 15, 15))               'input_3[0][0]',
                                                                      'input_3[0][0]']

     add (Add)                      (None, 15, 256)      0           ['input_3[0][0]',
                                                                      'multi_head_attention[0][0]']

     dense_4 (Dense)                (None, 15, 512)      131584      ['add[0][0]']

     dense_5 (Dense)                (None, 15, 256)      131328      ['dense_4[0][0]']

     dropout (Dropout)              (None, 15, 256)      0           ['dense_5[0][0]']

     add_1 (Add)                    (None, 15, 256)      0           ['add[0][0]',
                                                                      'dropout[0][0]']

     tf.math.multiply_8 (TFOpLambda  (None, 15, 256)     0           ['add_1[0][0]']
     )

    ===============================================================================================
    Total params: 526,080
    Trainable params: 526,080
    Non-trainable params: 0
    _______________________________________________________________________________________________
    """

    def __init__(self, config, **kwargs):
        super(MultiHeadAttentionModel, self).__init__(**kwargs)

        self.config = config

        self.mha1 = \
            tf.keras.layers.MultiHeadAttention(
                num_heads=self.config.num_heads,
                key_dim=self.config.key_dim,
            )

        self.add1 = \
            tf.keras.layers.Add()

        """
        self.layernorm1 = \
            tf.keras.layers.LayerNormalization(
                axis=-1, center=True, scale=True
            )
        """

        self.dense1 = \
            tf.keras.layers.Dense(
                units=config.hidden_dim * 2,
                activation='relu',
            )

        self.dense2 = \
            tf.keras.layers.Dense(
                units=config.hidden_dim,
                activation=None,
            )

        self.dropoout1 = tf.keras.layers.Dropout(rate=self.config.dropout_rate)

        self.add2 = tf.keras.layers.Add()

        """
        self.layernorm2 = \
            tf.keras.layers.LayerNormalization(
                axis=-1, center=True, scale=True
            )
        """

    @tf.function
    def call(self, inputs, mask=None, training=True):
        # inputs: (None,n,hidden_dim)=(None,15,256)
        # mask: (None,n)=(None,15), bool

        float_mask = \
            tf.expand_dims(
                tf.cast(mask, 'float32'),
                axis=-1
            )  # (None,n,1)

        attention_mask = tf.matmul(
            float_mask, float_mask, transpose_b=True
        )  # (None,n,n)

        attention_mask = tf.cast(attention_mask, 'bool')

        x, score = \
            self.mha1(
                query=inputs,
                key=inputs,
                value=inputs,
                attention_mask=attention_mask,
                return_attention_scores=True,
            )  # (None,n,hidden_dim),(None,num_heads,n,n)=(None,15,32),(None,2,15,15)

        x1 = self.add1([inputs, x])  # (None,n,hidden_dim)=(None,15,32)

        # x1 = self.layernorm1(x1)

        x2 = self.dense1(x1)  # (None,n,hidden_dim)=(None,15,512)

        x2 = self.dense2(x2)  # (None,n,hidden_dim)=(None,15,256)

        x2 = self.dropoout1(x2, training=training)

        features = self.add2([x1, x2])  # (None,n,hidden_dim)=(None,15,256)

        # features = self.layernorm2(features)

        broadcast_float_mask = \
            tf.expand_dims(
                tf.cast(mask, 'float32'),
                axis=-1
            )  # Add feature dim for broadcast, (None,n,1)=(1,15,1)

        features = features * broadcast_float_mask

        return features, score

    def build_graph(self, mask, idx):
        x = tf.keras.layers.Input(
            shape=(self.config.max_num_red_agents,
                   self.config.hidden_dim)
        )

        model = tf.keras.models.Model(
            inputs=[x],
            outputs=self.call(x, mask, training=True),
            name='mha_' + str(idx),
        )

        return model


class QLogitsModel(tf.keras.models.Model):
    """
    Very simple dense model, output is logits

    :param action_dim=5
    :param hidden_dim=256
    :param max_num_agents=15=n
    :return: (None,n,action_dim)=(None,15,5)

    Model: "q_net"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #
    =================================================================
     input_5 (InputLayer)        [(None, 15, 256)]         0

     time_distributed_14 (TimeDi  (None, 15, 768)          197376
     stributed)

     dropout_2 (Dropout)         (None, 15, 768)           0

     time_distributed_15 (TimeDi  (None, 15, 256)          196864
     stributed)

     time_distributed_16 (TimeDi  (None, 15, 5)            1285
     stributed)

     tf.math.multiply_10 (TFOpLa  (None, 15, 5)            0
     mbda)

    =================================================================
    Total params: 395,525
    Trainable params: 395,525
    Non-trainable params: 0
    _________________________________________________________________
    """

    def __init__(self, config, **kwargs):
        super(QLogitsModel, self).__init__(**kwargs)

        self.config = config

        self.dense1 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=self.config.hidden_dim * 3,
                    activation='relu',
                )
            )

        self.dropoout1 = tf.keras.layers.Dropout(rate=self.config.dropout_rate)

        self.dense2 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=self.config.hidden_dim,
                    activation='relu',
                )
            )

        self.dense3 = \
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=self.config.action_dim,
                    activation=None,
                )
            )

    @tf.function
    def call(self, inputs, mask, training=True):
        # inputs: (None,n,hidden_dim)=(None,15,256)
        # mask: (None,n)=(None,15), bool

        x1 = self.dense1(inputs)  # (None,n,hidden_dim*3)

        x1 = self.dropoout1(x1, training=training)

        x1 = self.dense2(x1)  # (None,n,hidden_dim)

        logits = self.dense3(x1)  # (None,n,action_dim)

        broadcast_float_mask = \
            tf.expand_dims(
                tf.cast(mask, 'float32'),
                axis=-1
            )  # Add feature dim for broadcast, (None,n,1)=(None,15,1)

        logits = logits * broadcast_float_mask

        return logits  # (1,15,5)

    def build_graph(self, mask):
        x = tf.keras.layers.Input(
            shape=(self.config.max_num_red_agents,
                   self.config.hidden_dim)
        )  # (None,n,256)

        model = tf.keras.models.Model(
            inputs=[x],
            outputs=self.call(x, mask),
            name='q_net'
        )

        return model


def main():
    dir_name = './models_architecture'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    config = Config()

    grid_size = config.grid_size
    ch = config.observation_channels
    n_frames = config.n_frames

    obs_shape = (grid_size, grid_size, ch * n_frames)

    max_num_agents = config.max_num_red_agents

    # Define alive_agents_ids & raw_obs
    alive_agents_ids = [0, 2]
    raw_obs = []

    for a in alive_agents_ids:
        obs_a = np.random.rand(obs_shape[0], obs_shape[1], obs_shape[2])
        raw_obs.append(obs_a)

    # Get padded_obs and mask
    padded_obs = make_padded_obs(max_num_agents, obs_shape, raw_obs)  # (1,n,g,g,ch*n_frames)

    mask = make_mask(alive_agents_ids, max_num_agents)  # (1,n)

    """ cnn_model """
    cnn = CNNModel(config=config)

    features_cnn = cnn(padded_obs, mask)  # Build, (1,n,hidden_dim)

    """ remove tf.function for summary """
    """
    cnn.build_graph(mask).summary()

    tf.keras.utils.plot_model(
        cnn.build_graph(mask),
        to_file=dir_name + '/cnn_model',
        show_shapes=True,
        show_layer_activations=True,
        show_dtype=True,
        dpi=96 * 3
    )
    """

    """ mha model """
    mha = MultiHeadAttentionModel(config=config)

    features_mha, score = mha(features_cnn[2], mask)  # Build, (None,n,hidden_dim),(1,num_heads,n,n)

    """ remove tf.function for summary """
    """
    idx = 1
    mha.build_graph(mask, idx).summary()

    tf.keras.utils.plot_model(
        mha.build_graph(mask, idx),
        to_file=dir_name + '/mha_model_' + str(idx),
        show_shapes=True,
        show_layer_activations=True,
        show_dtype=True,
        dpi=96 * 3
    )
    """

    """ q_model """
    q_net = QLogitsModel(config=config)

    q_logits = q_net(features_mha, mask)

    """ remove tf.function for summary """
    """
    q_net.build_graph(mask).summary()

    tf.keras.utils.plot_model(
        q_net.build_graph(mask),
        to_file=dir_name + '/q_logits_model',
        show_shapes=True,
        show_layer_activations=True,
        show_dtype=True,
        dpi=96 * 3
    )
    """


if __name__ == '__main__':
    main()
