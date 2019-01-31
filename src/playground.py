import tensorflow as tf
import os

from config import VALIDATION_TFRECORDS, N_VALIDATION
from data_loading import load_dataset

tf.enable_eager_execution()

MODEL_FOLDER = os.path.join('..', 'models', '2019-01-30_15-37-41')
N_EPOCHS = 50
BATCH_SIZE = 16
SHUFFLE_BUFFER = 1000
val_data = load_dataset(VALIDATION_TFRECORDS, BATCH_SIZE, SHUFFLE_BUFFER)

model = tf.keras.models.load_model(os.path.join(MODEL_FOLDER, 'cnn_model.h5'))
# the following line is necessary because eager execution doesn't support keras optimizers
# and keras saver doesn't support tf optimizers
model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

y_pred = model.predict(val_data, steps=N_VALIDATION // BATCH_SIZE)
auc = tf.metrics.auc(val_data, y_pred, curve='ROC', summation_method='trapezoidal')
model.evaluate(val_data, steps=N_VALIDATION // BATCH_SIZE)
