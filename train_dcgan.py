import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from DCGAN import DCGAN
from Generator import Generator
import datetime
import os
from ImageWriterCallback import ImageWriterCallback
from ModelSaveCallback import ModelSaveCallback

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
# generator = Generator(images=x_train, batch_size=16, conditions=y_train)

x_train=np.load('./data/celeba.npy')
generator = Generator(images=x_train, batch_size=64, conditions=None)

# plt.figure()
# for i in range(4*4):
#         plt.subplot(4,4,i+1)
#         plt.imshow(generator[0][0][i],vmin=-1,vmax=1)#,cmap='gray')
# plt.show()


dcgan = DCGAN(conditional=False)

time_stamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
log_dir = './logs/' + time_stamp
os.makedirs(log_dir)
os.makedirs(log_dir + '/images')

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir)
image_writer_callback = ImageWriterCallback(log_dir + '/images')
model_save_callback = ModelSaveCallback(log_dir)

# dcgan.load_models('./logs/2021_09_19_01_01_51/')
dcgan.print_model_summary()
dcgan.compile(run_eagerly=False)
history = dcgan.fit(generator, epochs=75, callbacks=[tensorboard_callback, image_writer_callback,model_save_callback])