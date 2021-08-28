import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


class ImageWriterCallback(tf.keras.callbacks.Callback):

    def __init__(self,log_dir):
        self.log_dir=log_dir
        super(ImageWriterCallback,self).__init__()
        self.step=0
        self.tensorboard_writer = tf.summary.create_file_writer(self.log_dir)
        self.amount_of_rows_cols=5
        self.seed=tf.random.set_seed=1


    def on_epoch_end(self, epoch, logs=None):
        self.noise = tf.random.normal(shape=(self.amount_of_rows_cols**2, self.model.latent_dim), mean=0, stddev=1,seed=self.seed)

        fake_images=self.model.generate_images(self.noise)

        _,height,width,depth=fake_images.shape
        image_collection=np.zeros((self.amount_of_rows_cols*height,self.amount_of_rows_cols*width,1))
        n=0
        for i in range(self.amount_of_rows_cols):
            for j in range(self.amount_of_rows_cols):
                image_collection[i*height:(i+1)*height,j*width:(j+1)*width,:]=fake_images[n]
                n+=1


        # plt.figure()
        # plt.imshow(image_collection,cmap='gray',vmin=-1,vmax=1)
        # plt.show()
        with self.tensorboard_writer.as_default():
            tf.summary.image('generated images',image_collection.reshape((-1,self.amount_of_rows_cols*height,self.amount_of_rows_cols*width,1)),max_outputs=1,step=self.step)
        self.step+=1