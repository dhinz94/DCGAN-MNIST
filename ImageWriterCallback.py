import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import cv2

class ImageWriterCallback(tf.keras.callbacks.Callback):

    def __init__(self,tensorboard_log_dir):
        self.log_dir=tensorboard_log_dir
        super(ImageWriterCallback,self).__init__()
        self.tensorboard_writer = tf.summary.create_file_writer(self.log_dir)
        self.amount_of_rows_cols=5
        self.gif_images=[]



    def on_epoch_end(self, epoch, logs=None):
        tf.random.set_seed(5)
        self.noise = tf.random.normal(shape=(self.amount_of_rows_cols**2, self.model.latent_dim), mean=0, stddev=1,seed=1)

        fake_images=self.model.generate_images(self.noise)

        _,height,width,depth=fake_images.shape
        generated_images=np.zeros((self.amount_of_rows_cols*height,self.amount_of_rows_cols*width,1))

        n=0
        for i in range(self.amount_of_rows_cols):
            for j in range(self.amount_of_rows_cols):
                generated_images[i*height:(i+1)*height,j*width:(j+1)*width,:]=fake_images[n]
                n+=1

        # plt.figure()
        # plt.imshow(generated_images,cmap='gray',vmin=-1,vmax=1)
        # plt.show()
        self.gif_images.append(((generated_images+1)*127.5).astype('uint8'))
        cv2.imwrite(os.path.join(self.log_dir,'generated_images.png'),self.gif_images[-1])

        with self.tensorboard_writer.as_default():
            tf.summary.image('generated images',generated_images.reshape((-1,self.amount_of_rows_cols*height,self.amount_of_rows_cols*width,1)),max_outputs=1,step=epoch)

    def on_train_end(self, epoch, logs=None):
        self.gif_images=[Image.fromarray(image[:,:,0]) for image in self.gif_images]
        self.gif_images[0].save(os.path.join(self.log_dir,'train.gif'), save_all=True, append_images=self.gif_images[1:], duration=100, loop=0)