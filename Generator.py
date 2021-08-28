import tensorflow as tf
import numpy as np

class Generator(tf.keras.utils.Sequence):

    def __init__(self,images,batch_size):
        self.batch_size = batch_size
        self.images=images
        p=np.random.permutation(len(self.images))
        self.images=self.images[p]

    def __len__(self):
        return np.ceil(len(self.images)/self.batch_size).astype('int')

    def __getitem__(self, i):
        images_batch=self.images[i*self.batch_size:(i+1)*self.batch_size]
        images_batch=images_batch.astype('float32')/127.5-1
        return images_batch

    def on_epoch_end(self):
        p = np.random.permutation(len(self.images))
        self.images = self.images[p]

