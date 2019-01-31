import os

import cv2
import tensorflow as tf
import models
from config import TRAIN_TFRECORDS, VALIDATION_TFRECORDS, N_TRAIN, N_VALIDATION
from data_loading import load_dataset

tf.enable_eager_execution()

MODEL_FOLDER = os.path.join('..', 'models', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
N_EPOCHS = 50
BATCH_SIZE = 16
SHUFFLE_BUFFER = 1000
train_data = load_dataset(TRAIN_TFRECORDS, BATCH_SIZE, SHUFFLE_BUFFER)
val_data = load_dataset(VALIDATION_TFRECORDS, BATCH_SIZE, SHUFFLE_BUFFER)

model = models.first_attempt(print_summary=True)
# cv2.imshow('image', train_iterator.get_next()[0][0].numpy())
# cv2.waitKey(0)
# cv2.destroyAllWindows()


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min', baseline=None),
    tf.keras.callbacks.TensorBoard(log_dir=MODEL_FOLDER, histogram_freq=1)
]

model.fit(
    train_data,
    epochs=N_EPOCHS,
    steps_per_epoch=N_TRAIN // BATCH_SIZE,
    validation_data=val_data,
    validation_steps=N_VALIDATION // BATCH_SIZE,
    callbacks=callbacks
)

model.save(os.path.join(MODEL_FOLDER, 'cnn_model.h5'))
