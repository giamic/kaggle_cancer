import tensorflow as tf

from config import IMG_SIZE


def simple_dense(print_summary=True):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)))
    model.add(tf.keras.layers.Dense(1, 'sigmoid'))

    model.compile(
        optimizer=tf.train.AdamOptimizer(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    if print_summary:
        model.summary()
    return model


def first_attempt(print_summary=True):
    model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Dropout(rate=0.1))
    model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D())
    # model.add(tf.keras.layers.Dropout(rate=0.1))
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D())
    # model.add(tf.keras.layers.Dropout(rate=0.1))
    model.add(tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1, 'sigmoid'))

    model.compile(
        optimizer=tf.train.AdamOptimizer(),  # doesn't work perfectly with keras saver
        # optimizer=tf.keras.optimizers.Adam(),  # doesn't work at all with tf eager execution
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    if print_summary:
        model.summary()
    return model
