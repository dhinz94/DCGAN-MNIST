import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from DCGAN import DCGAN
from Generator import Generator
import datetime
import os
from ImageWriterCallback import ImageWriterCallback

(x_train, y_train), (x_test, y_test)=tf.keras.datasets.mnist.load_data()

generator=Generator(images=x_train,batch_size=16)

# plt.figure()
# for i in range(4*4):
#         plt.subplot(4,4,i+1)
#         plt.imshow(generator[0][i],cmap='gray',vmin=-1,vmax=1)
# plt.show()


dcgan=DCGAN()
dcgan.print_model_summary()
dcgan.compile()

time_stamp=datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
log_dir='./logs/'+time_stamp
os.makedirs(log_dir)
os.makedirs(log_dir+'/images')

tensorboard_callback=tf.keras.callbacks.TensorBoard(log_dir)
image_writer_callback=ImageWriterCallback(log_dir+'/images')
dcgan.fit(generator,epochs=50,callbacks=[tensorboard_callback,image_writer_callback])


