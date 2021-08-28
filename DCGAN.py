import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Conv2D, InputLayer, BatchNormalization, Activation, Conv2DTranspose, Dropout, GlobalAvgPool2D, Dense, Reshape, Input, Flatten
import numpy as np
import tensorflow.keras.backend as K
import os


class DCGAN(tf.keras.Model):
    def __init__(self):
        super(DCGAN, self).__init__()
        self.dropout_rate = 0.3
        self.latent_dim = 100

        self.generator = self.get_generator()
        self.discriminator = self.get_discriminator()

        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)

    def save_models(self, path):
        self.generator.save(os.path.join(path, 'generator.h5'))
        self.discriminator.save(os.path.join(path, 'discriminator.h5'))

    def print_model_summary(self):
        print('generator:')
        print(self.generator.summary())
        print('discriminator:')
        print(self.discriminator.summary())

    def gen_loss(self, discriminator_output_fake):
        loss = K.mean(K.square(discriminator_output_fake - tf.random.normal(tf.shape(discriminator_output_fake), mean=0.85, stddev=0.05)))
        return loss

    def dis_loss(self, discriminator_output_real, discriminator_output_fake):
        fake_loss = K.mean(K.square(discriminator_output_fake - tf.random.normal(tf.shape(discriminator_output_fake), mean=0.15, stddev=0.05)))
        real_loss = K.mean(K.square(discriminator_output_real - tf.random.normal(tf.shape(discriminator_output_real), mean=0.85, stddev=0.05)))
        loss = (fake_loss + real_loss) / 2
        return loss

    def downsampling_block(self, input_tensor, filters, dropout=False):
        x = Conv2D(filters, kernel_size=(3, 3), strides=2, padding='same', activation=None)(input_tensor)
        x = BatchNormalization()(x)
        x = Activation(tf.nn.leaky_relu)(x)
        if dropout:
            x = Dropout(self.dropout_rate)(x)

        x = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same', activation=None)(x)
        x = BatchNormalization()(x)
        x = Activation(tf.nn.leaky_relu)(x)
        if dropout:
            x = Dropout(self.dropout_rate)(x)

        return x

    def upsampling_block(self, input_tensor, filters, dropout=False):

        x = Conv2DTranspose(filters, kernel_size=(3, 3), strides=2, padding='same', activation=None)(input_tensor)
        x = BatchNormalization()(x)
        x = Activation(tf.nn.leaky_relu)(x)
        if dropout:
            x = Dropout(self.dropout_rate)(x)

        x = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same', activation=None)(x)
        x = BatchNormalization()(x)
        x = Activation(tf.nn.leaky_relu)(x)
        if dropout:
            x = Dropout(self.dropout_rate)(x)

        return x

    def get_generator(self):

        end_filters = 16
        inp = Input(shape=(self.latent_dim))
        x = Dense(7 * 7 * end_filters * 2, activation=None)(inp)
        x = Reshape(target_shape=(7, 7, end_filters * 2))(x)
        x = self.upsampling_block(x, end_filters * 2, dropout=False)  # 14x14
        x = self.upsampling_block(x, end_filters * 1, dropout=False)  # 28x28
        x = Conv2D(1, kernel_size=(1, 1), padding='same', activation=None)(x)
        x = Activation('tanh')(x)

        model = tf.keras.Model(inputs=inp, outputs=x)

        return model

    def get_discriminator(self):
        start_filters = 16

        inp = Input(shape=(28, 28, 1))
        x = self.downsampling_block(inp, start_filters, dropout=True)  # 14x14
        x = self.downsampling_block(x, start_filters * 2, dropout=True)  # 7x7
        x = Flatten()(x)
        x = Dense(100, activation=tf.nn.leaky_relu)(x)
        x = Dense(1, activation=None)(x)
        x = Activation(tf.nn.sigmoid)(x)

        model = tf.keras.Model(inputs=inp, outputs=x)

        return model

    def compile(self):
        super(DCGAN, self).compile()
        pass


    def generate_images(self, noise):
        return self.generator(noise)

    @tf.function
    def train_step(self, real_images):

        with tf.GradientTape() as discriminator_tape, tf.GradientTape() as generator_tape:
            noise = tf.random.normal(shape=(tf.shape(real_images)[0], self.latent_dim), mean=0, stddev=1)
            fake_images = self.generator(noise)
            discriminator_output_real = self.discriminator(real_images)
            discriminator_output_fake = self.discriminator(fake_images)

            generator_loss = self.gen_loss(discriminator_output_fake)
            discriminator_loss = self.dis_loss(discriminator_output_real, discriminator_output_fake)

        generator_gradients = generator_tape.gradient(generator_loss, self.generator.trainable_weights)
        discriminator_gradients = discriminator_tape.gradient(discriminator_loss, self.discriminator.trainable_weights)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_weights))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_weights))

        loss_dict = {'generator_loss': generator_loss, 'discriminator_loss': discriminator_loss}

        return loss_dict
