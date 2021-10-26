import tensorflow as tf

class ModelSaveCallback(tf.keras.callbacks.Callback):

    def __init__(self, tensorboard_log_dir):
        self.log_dir = tensorboard_log_dir
        super(ModelSaveCallback, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        self.model.save_models(self.log_dir)

