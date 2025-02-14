import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Conv2D, InputLayer, BatchNormalization, Activation, Conv2DTranspose, Dropout, AveragePooling2D, GlobalAvgPool2D, Dense, Reshape, Input, Flatten, Concatenate, UpSampling2D, Add
import numpy as np
import tensorflow.keras.backend as K
import os


class InstanceNormalization(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)

    def call(self, inputs):
        mean = K.mean(inputs, axis=[1, 2], keepdims=True)
        std = K.std(inputs, axis=[1, 2], keepdims=True) + 1e-8
        return (inputs - mean) / std

    def get_config(self):
        base_config = super(InstanceNormalization, self).get_config()
        return base_config


class DCGAN(tf.keras.Model):
    def __init__(self, conditional=False):
        super(DCGAN, self).__init__()
        self.dropout_rate = 0.2
        self.latent_dim = 100
        self.is_conditional = conditional
        # self.initializer = tf.random_normal_initializer(0, 1e-2)  # 2e-2 for BN for CelebA, 1e-3 for MNIST and BN
        self.initializer = None
        self.noise_multiplier = 0.0
        self.discriminator_steps = 1
        self.gp_weight = 1

        self.generator = self.get_big_generator()
        self.discriminator = self.get_big_discriminator()

        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4, beta_1=0.5)
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4, beta_1=0.5)

    def save_models(self, path):
        self.generator.save(os.path.join(path, 'generator.h5'))
        self.discriminator.save(os.path.join(path, 'discriminator.h5'))

    def load_models(self, path):
        self.discriminator = tf.keras.models.load_model(os.path.join(path, 'discriminator.h5'))
        self.generator = tf.keras.models.load_model(os.path.join(path, 'generator.h5'))

    def print_model_summary(self):
        print('generator:')
        print(self.generator.summary())
        print('discriminator:')
        print(self.discriminator.summary())

    def gen_loss(self, discriminator_output_fake):

        # loss = K.mean(-K.log((discriminator_output_fake) + 1e-8))

        # loss = K.mean(K.square(discriminator_output_fake - tf.random.normal(tf.shape(discriminator_output_fake), mean=1.00, stddev=0.2)))

        # loss = -K.mean(discriminator_output_fake)

        loss = K.mean(tf.nn.softplus(-discriminator_output_fake))
        return loss

    def dis_loss(self, discriminator_output_real, discriminator_output_fake):
        # fake_loss = K.mean(-K.log((1 - discriminator_output_fake) + 1e-8))
        # real_loss = K.mean(-K.log((discriminator_output_real) + 1e-8))

        # fake_loss = K.mean(K.square(discriminator_output_fake - tf.random.normal(tf.shape(discriminator_output_fake), mean=0.00, stddev=0.2)))
        # real_loss = K.mean(K.square(discriminator_output_real - tf.random.normal(tf.shape(discriminator_output_real), mean=1.00, stddev=0.2)))
        # loss = (fake_loss + real_loss) / 2

        # loss = K.mean(discriminator_output_fake) - K.mean(discriminator_output_real)

        loss = K.mean(tf.nn.softplus(discriminator_output_fake)) + K.mean(tf.nn.softplus(-discriminator_output_real))
        return loss

    def downsampling_block(self, input_tensor, filters, dropout=False):
        x = Conv2D(filters, kernel_size=(3, 3), strides=2, padding='same', activation=None, kernel_initializer=self.initializer, use_bias=False)(input_tensor)
        x = BatchNormalization()(x)
        # x = InstanceNormalization()(x)
        x = Activation(tf.nn.leaky_relu)(x)
        if dropout:
            x = Dropout(self.dropout_rate)(x)

        # x = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same', activation=None, kernel_initializer=self.initializer, use_bias=False)(x)
        # # x = BatchNormalization()(x)
        # # x = InstanceNormalization()(x)
        # x = Activation(tf.nn.leaky_relu)(x)
        # if dropout:
        #     x = Dropout(self.dropout_rate)(x)

        return x

    def residual_block(self, input_tensor, filters, dropout=False):
        x = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same', activation=None, kernel_initializer=self.initializer, use_bias=False)(input_tensor)
        x = BatchNormalization()(x)
        # x = InstanceNormalization()(x)
        x = Activation(tf.nn.leaky_relu)(x)
        if dropout:
            x = Dropout(self.dropout_rate)(x)

        x = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same', activation=None, kernel_initializer=self.initializer, use_bias=False)(x)
        x = BatchNormalization()(x)
        # x = InstanceNormalization()(x)
        x = Activation(tf.nn.leaky_relu)(x)
        if dropout:
            x = Dropout(self.dropout_rate)(x)

        x = Add()([input_tensor, x])

        return x

    def upsampling_block(self, input_tensor, filters, dropout=False):

        x = Conv2DTranspose(filters, kernel_size=(3, 3), strides=2, padding='same', activation=None, kernel_initializer=self.initializer, use_bias=False)(input_tensor)
        x = BatchNormalization()(x)
        # x = InstanceNormalization()(x)
        x = Activation(tf.nn.leaky_relu)(x)
        if dropout:
            x = Dropout(self.dropout_rate)(x)

        x = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same', activation=None, kernel_initializer=self.initializer, use_bias=False)(x)
        x = BatchNormalization()(x)
        # x = InstanceNormalization()(x)
        x = Activation(tf.nn.leaky_relu)(x)
        if dropout:
            x = Dropout(self.dropout_rate)(x)

        return x

    def conv_block(self, input_tensor, filters, dropout=False):
        x = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same', activation=None, kernel_initializer=self.initializer, use_bias=False)(input_tensor)
        x = BatchNormalization()(x)
        # x = InstanceNormalization()(x)
        x = Activation(tf.nn.leaky_relu)(x)
        if dropout:
            x = Dropout(self.dropout_rate)(x)

        x = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same', activation=None, kernel_initializer=self.initializer, use_bias=False)(x)
        x = BatchNormalization()(x)
        # x = InstanceNormalization()(x)
        x = Activation(tf.nn.leaky_relu)(x)
        if dropout:
            x = Dropout(self.dropout_rate)(x)

        return x

    def get_generator(self):

        end_filters = 64
        inp_latent = Input(shape=(self.latent_dim))
        if self.is_conditional:
            inp_condition = Input(shape=(10))
            x = Concatenate(axis=-1)([inp_latent, inp_condition])
            x = Dense(7 * 7 * end_filters * 2, activation=None, kernel_initializer=self.initializer, use_bias=False)(x)
            # x = BatchNormalization()(x)
            x = Activation(tf.nn.leaky_relu)(x)
        else:
            x = Dense(7 * 7 * end_filters * 2, activation=tf.nn.leaky_relu, kernel_initializer=self.initializer, use_bias=False)(inp_latent)
            # x = BatchNormalization()(x)
            x = Activation(tf.nn.leaky_relu)(x)

        x = Reshape(target_shape=(7, 7, end_filters * 2))(x)
        x = self.upsampling_block(x, end_filters * 2, dropout=False)  # 14x14
        x = self.upsampling_block(x, end_filters * 1, dropout=False)  # 28x28
        x = Conv2D(1, kernel_size=(1, 1), padding='same', activation=None)(x)
        x = Activation('tanh')(x)

        if self.is_conditional:
            model = tf.keras.Model(inputs=[inp_latent, inp_condition], outputs=x)
        else:
            model = tf.keras.Model(inputs=inp_latent, outputs=x)

        return model

    def get_discriminator(self):
        start_filters = 64

        inp_image = Input(shape=(28, 28, 1))
        if self.is_conditional:
            inp_condition = Input(shape=(10,))
            x = Dense(28 * 28 * 1, activation=None, kernel_initializer=self.initializer, use_bias=False)(inp_condition)
            x = BatchNormalization()(x)
            x = Activation(tf.nn.leaky_relu)(x)
            x = Reshape(target_shape=(28, 28, 1))(x)
            x = Concatenate(axis=-1)([inp_image, x])
            x = self.downsampling_block(x, start_filters, dropout=True)  # 14x14
        else:
            x = self.downsampling_block(inp_image, start_filters, dropout=True)  # 14x14

        x = self.downsampling_block(x, start_filters * 2, dropout=True)  # 7x7
        x = Flatten()(x)
        x = Dense(100, activation=tf.nn.leaky_relu, kernel_initializer=self.initializer)(x)
        x = Dense(1, activation=None, kernel_initializer=self.initializer)(x)
        x = Activation(tf.nn.sigmoid)(x)

        if self.is_conditional:
            model = tf.keras.Model(inputs=[inp_image, inp_condition], outputs=x)
        else:
            model = tf.keras.Model(inputs=inp_image, outputs=x)

        return model

    def get_big_generator(self):
        end_filters = 32
        inp_latent = Input(shape=(self.latent_dim))
        interpolation = 'bilinear'

        x = Dense(8 * 8 * end_filters * 8, activation=tf.nn.leaky_relu, kernel_initializer=self.initializer, use_bias=True)(inp_latent)

        x = Reshape(target_shape=(8, 8, end_filters * 8))(x)
        x = BatchNormalization()(x)
        # x = InstanceNormalization()(x)
        x = Activation(tf.nn.leaky_relu)(x)

        x8 = Conv2D(3, kernel_size=(3, 3), padding='same', activation=None, kernel_initializer=self.initializer)(x)
        # x8 = Activation('tanh')(x8)

        # conv upsampling
        x = self.upsampling_block(x, end_filters * 4, dropout=False)  # 16x16
        x16 = Conv2D(3, kernel_size=(3, 3), padding='same', activation=None, kernel_initializer=self.initializer)(x)
        # x16 = Activation('tanh')(x16)
        # x = self.residual_block(x, end_filters*8, dropout=True)

        x = self.upsampling_block(x, end_filters * 2, dropout=False)  # 32x32
        x32 = Conv2D(3, kernel_size=(3, 3), padding='same', activation=None, kernel_initializer=self.initializer)(x)
        # x32 = Activation('tanh')(x32)
        # x = self.residual_block(x, end_filters*4, dropout=True)

        x = self.upsampling_block(x, end_filters * 1, dropout=False)  # 64x64
        x64 = Conv2D(3, kernel_size=(3, 3), padding='same', activation=None, kernel_initializer=self.initializer)(x)
        # x64 = Activation('tanh')(x64)

        x = self.upsampling_block(x, end_filters * 1, dropout=False)  # 128x128
        x128 = Conv2D(3, kernel_size=(3, 3), padding='same', activation=None, kernel_initializer=self.initializer)(x)
        # x128 = Activation('tanh')(x128)

        # # traditional upsampling
        # x = UpSampling2D(interpolation=interpolation)(x)  # 16x16
        # x = self.conv_block(x, end_filters * 4, dropout=False)
        # x16 = Conv2D(3, kernel_size=(3, 3), padding='same', activation=None, kernel_initializer=self.initializer)(x)
        # # x16 = Activation('tanh')(x16)
        #
        # x = UpSampling2D(interpolation=interpolation)(x)  # 32x32
        # x = self.conv_block(x, end_filters * 2, dropout=False)
        # x32 = Conv2D(3, kernel_size=(3, 3), padding='same', activation=None, kernel_initializer=self.initializer)(x)
        # # x32 = Activation('tanh')(x32)
        #
        # x = UpSampling2D(interpolation=interpolation)(x)  # 64x64
        # x = self.conv_block(x, end_filters * 1, dropout=False)
        # x64 = Conv2D(3, kernel_size=(3, 3), padding='same', activation=None, kernel_initializer=self.initializer)(x)
        # # x64 = Activation('tanh')(x64)

        # # for 128
        # x8 = UpSampling2D(interpolation=interpolation, size=(16, 16))(x8)
        # x16 = UpSampling2D(interpolation=interpolation, size=(8, 8))(x16)
        # x32 = UpSampling2D(interpolation=interpolation, size=(4, 4))(x32)
        # x64 = UpSampling2D(interpolation=interpolation, size=(2, 2))(x64)
        # output = Add()([x8, x16, x32, x64, x128])

        # # for 64
        # x8 = UpSampling2D(interpolation=interpolation, size=(8, 8))(x8)
        # x16 = UpSampling2D(interpolation=interpolation, size=(4, 4))(x16)
        # x32 = UpSampling2D(interpolation=interpolation, size=(2, 2))(x32)
        # output = Add()([x8, x16, x32, x64])

        output = Activation('tanh')(x64)

        model = tf.keras.Model(inputs=inp_latent, outputs=output)

        return model

    def get_big_discriminator(self):
        start_filters = 32

        # inp_image = Input(shape=(128, 128, 3))
        # inp_image_64 = AveragePooling2D(pool_size=(2, 2))(inp_image)
        # inp_image_32 = AveragePooling2D(pool_size=(4, 4))(inp_image)
        # inp_image_16 = AveragePooling2D(pool_size=(8, 8))(inp_image)
        # inp_image_8 = AveragePooling2D(pool_size=(16, 16))(inp_image)
        # inp_image_4 = AveragePooling2D(pool_size=(32, 32))(inp_image)

        inp_image = Input(shape=(64, 64, 3))
        inp_image_32 = AveragePooling2D(pool_size=(2, 2))(inp_image)
        inp_image_16 = AveragePooling2D(pool_size=(4, 4))(inp_image)
        inp_image_8 = AveragePooling2D(pool_size=(8, 8))(inp_image)
        inp_image_4 = AveragePooling2D(pool_size=(16, 16))(inp_image)

        # x = self.downsampling_block(inp_image, start_filters, dropout=True)  # 64x64
        # x = self.residual_block(x, start_filters, dropout=True)
        # x = Concatenate()([x, inp_image_64])

        x = self.downsampling_block(inp_image, start_filters, dropout=True)  # 32x32
        x = self.residual_block(x, start_filters, dropout=True)
        x = Concatenate()([x, inp_image_32])

        x = self.downsampling_block(x, start_filters * 2, dropout=True)  # 16x16
        x = self.residual_block(x, start_filters * 2, dropout=True)
        x = Concatenate()([x, inp_image_16])

        x = self.downsampling_block(x, start_filters * 4, dropout=True)  # 8x8
        x = self.residual_block(x, start_filters * 4, dropout=True)
        x = Concatenate()([x, inp_image_8])

        x = self.downsampling_block(x, start_filters * 8, dropout=True)  # 4x4
        x = self.residual_block(x, start_filters * 8, dropout=True)
        x = Concatenate()([x, inp_image_4])

        x = Flatten()(x)
        x = Dense(100, activation=tf.nn.leaky_relu, kernel_initializer=self.initializer)(x)
        x = Dense(1, activation=None, kernel_initializer=self.initializer)(x)
        # x = Conv2D(1, kernel_size=(3, 3), padding='same', activation=None, kernel_initializer=self.initializer)(x)
        # x = Activation(tf.nn.sigmoid)(x) #comment out for linear activation

        model = tf.keras.Model(inputs=inp_image, outputs=x)

        return model

    def compile(self, **kwargs):
        super(DCGAN, self).compile(**kwargs)

    def call(self, noise):
        pass

    def generate_images(self, noise, condition=None):
        if self.is_conditional:
            generated_images = self.generator([noise, condition])
        else:
            generated_images = self.generator(noise)
        return generated_images

    def train_discriminator(self, real_images, conditions=None):
        with tf.GradientTape() as discriminator_tape, tf.GradientTape() as gp_tape:
            discriminator_tape.watch(real_images)
            gp_tape.watch(real_images)

            noise = tf.random.normal(shape=(tf.shape(real_images)[0], self.latent_dim), mean=0, stddev=1)
            if self.is_conditional:

                fake_images = self.generator([noise, conditions])
                fake_images = K.clip(fake_images + self.noise_multiplier * tf.random.normal(tf.shape(fake_images), 0, 1), -1, 1)
                real_images = K.clip(real_images + self.noise_multiplier * tf.random.normal(tf.shape(fake_images), 0, 1) - 1, 1)
                discriminator_output_real = self.discriminator([real_images, conditions])
                discriminator_output_fake = self.discriminator([fake_images, conditions])
            else:
                fake_images = self.generator(noise)

                discriminator_output_real = self.discriminator(real_images)
                discriminator_output_fake = self.discriminator(fake_images)

                epsilon = K.random_uniform(shape=[tf.shape(real_images)[0], 1, 1, 1], minval=0, maxval=1)
                interpolated_images = epsilon * real_images + (1 - epsilon) * fake_images
                discriminator_output_interpolated_images = self.discriminator(interpolated_images)

            dis_gp = gp_tape.gradient(discriminator_output_interpolated_images, interpolated_images)
            dis_gp = K.sqrt(K.sum(K.square(dis_gp), axis=[1, 2, 3]))  # norm of gradients for every image
            dis_gp = K.mean(K.square(dis_gp - 1 * K.ones_like(dis_gp)))  # force norm of every gradient to 1

            discriminator_loss = self.dis_loss(discriminator_output_real, discriminator_output_fake) + self.gp_weight * dis_gp
        discriminator_gradients = discriminator_tape.gradient(discriminator_loss, self.discriminator.trainable_weights)
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_weights))

        return discriminator_loss, dis_gp

    def train_generator(self, real_images, conditions=None):
        with tf.GradientTape() as generator_tape:
            noise = tf.random.normal(shape=(tf.shape(real_images)[0], self.latent_dim), mean=0, stddev=1)
            if self.is_conditional:
                fake_images = self.generator([noise, conditions])
                discriminator_output_fake = self.discriminator([fake_images, conditions])
            else:
                fake_images = self.generator(noise)
                discriminator_output_fake = self.discriminator(fake_images)

            generator_loss = self.gen_loss(discriminator_output_fake)
        generator_gradients = generator_tape.gradient(generator_loss, self.generator.trainable_weights)
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_weights))
        return generator_loss

    @tf.function
    def train_step(self, data):
        if self.is_conditional:
            real_images = data[0]
            conditions = data[1]
        else:
            real_images = data
            real_images = tf.convert_to_tensor(real_images)

        for i in range(self.discriminator_steps):
            discriminator_loss, dis_gp = self.train_discriminator(real_images)
        generator_loss = self.train_generator(real_images)

        loss_dict = {'generator_loss': generator_loss, 'discriminator_loss': discriminator_loss, 'gradient_penalty': dis_gp}

        return loss_dict
