import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import cv2


class ImageWriterCallback(tf.keras.callbacks.Callback):

    def __init__(self, tensorboard_log_dir):
        self.log_dir = tensorboard_log_dir
        super(ImageWriterCallback, self).__init__()
        self.tensorboard_writer = tf.summary.create_file_writer(self.log_dir)
        self.amount_of_rows_cols = 10
        self.gif_images = []

    def on_epoch_begin(self, epoch, logs=None):
        tf.random.set_seed(5)
        self.noise = tf.random.normal(shape=(self.amount_of_rows_cols ** 2, self.model.latent_dim), mean=0, stddev=1, seed=1)
        if self.model.is_conditional:
            self.conditions = np.eye(10)[np.tile(np.arange(0, 10).reshape(-1, 1), (1, 10)).reshape(-1)]
            fake_images = self.model.generate_images(noise=self.noise, condition=self.conditions)

        else:
            fake_images = self.model.generate_images(self.noise)

        _, height, width, depth = fake_images.shape
        generated_images = np.zeros((self.amount_of_rows_cols * height, self.amount_of_rows_cols * width, depth))

        n = 0
        for i in range(self.amount_of_rows_cols):
            for j in range(self.amount_of_rows_cols):
                generated_images[i * height:(i + 1) * height, j * width:(j + 1) * width, :] = fake_images[n]
                n += 1

        generated_images = ((generated_images + 1) * 127.5).astype('uint8')
        if depth==3:
            generated_images=cv2.cvtColor(generated_images,cv2.COLOR_BGR2RGB)

        # plt.figure()
        # plt.imshow(generated_images,cmap='gray',vmin=-1,vmax=1)
        # plt.show()

        self.gif_images.append(generated_images)
        cv2.imwrite(os.path.join(self.log_dir, 'generated_images.png'), self.gif_images[-1])

        with self.tensorboard_writer.as_default():
            tf.summary.image('generated images', np.flip(generated_images.reshape((-1, self.amount_of_rows_cols * height, self.amount_of_rows_cols * width, depth)),axis=-1), max_outputs=1, step=epoch)

    def on_train_end(self, epoch, logs=None):
        self.gif_images = [Image.fromarray(np.flip(image[:, :, :],axis=-1)) for image in self.gif_images]
        self.gif_images[0].save(os.path.join(self.log_dir, 'train.gif'), save_all=True, append_images=self.gif_images[1:], duration=100, loop=0)
