import tensorflow as tf
import numpy as np


class Generator(tf.keras.utils.Sequence):

    def __init__(self, images, batch_size, conditions=None):
        self.batch_size = batch_size
        self.images = images
        self.is_conditional = not isinstance(conditions, type(None))
        if self.is_conditional:
            self.conditions = conditions

        p = np.random.permutation(len(self.images))
        self.images = self.images[p]
        if self.is_conditional:
            self.conditions = self.conditions[p]

    def __len__(self):
        return np.ceil(len(self.images) / self.batch_size).astype('int')

    def __getitem__(self, i):
        images_batch = self.images[i * self.batch_size:(i + 1) * self.batch_size]
        images_batch = images_batch.astype('float32') / 127.5 - 1
        if self.is_conditional:
            conditions_batch = self.conditions[i * self.batch_size:(i + 1) * self.batch_size]
            conditions_batch = np.eye(10)[conditions_batch].astype('float32')
            return (images_batch, conditions_batch)
        else:
            return (images_batch)

    def on_epoch_end(self):
        p = np.random.permutation(len(self.images))
        self.images = self.images[p]
        if self.is_conditional:
            self.conditions = self.conditions[p]
