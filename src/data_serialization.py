import os
import random

import cv2
import pandas as pd
import tensorflow as tf

from config import LABELS_PATH, TRAIN_FOLDER, VALIDATION_FOLDER, IMG_SIZE, TRAIN_TFRECORDS, VALIDATION_TFRECORDS


def create_validation_set():
    import numpy as np
    os.makedirs(VALIDATION_FOLDER)
    images = os.listdir(TRAIN_FOLDER)
    file_paths = [os.path.join(TRAIN_FOLDER, i) for i in images]
    for fp in file_paths:
        if np.random.random() > 0.90:
            destination = os.path.join(VALIDATION_FOLDER, os.path.basename(fp))
            os.rename(fp, destination)
    return


def create_tfrecords(data_path, labels_path, output_path):
    if os.path.isfile(output_path):
        raise PermissionError("The output file already exists, exiting to avoid data loss.")

    labels_csv = pd.read_csv(LABELS_PATH)
    labels = dict(zip(labels_csv['id'].values, labels_csv['label'].values))

    file_names = os.listdir(data_path)
    random.seed(24061987)
    random.shuffle(file_names)
    n, N = 0, len(file_names)

    with tf.python_io.TFRecordWriter(output_path) as writer:
        for fn in file_names:
            if n % 100 == 0:
                print("Image {} of {}".format(n, N))
            img = cv2.imread(os.path.join(data_path, fn))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            x = img.flatten()
            assert len(x) == IMG_SIZE[0] * IMG_SIZE[1]
            label = labels[fn.split('.')[0]]
            example = tf.train.Example()
            example.features.feature["label"].int64_list.value.append(label)
            example.features.feature["x"].float_list.value.extend(x)
            writer.write(example.SerializeToString())
            n += 1
    return


if __name__ == '__main__':
    try:
        create_validation_set()
    except IOError:
        print("Validation data already exists, ignoring train-validation split.")
        pass

    create_tfrecords(TRAIN_FOLDER, LABELS_PATH, TRAIN_TFRECORDS)
    create_tfrecords(VALIDATION_FOLDER, LABELS_PATH, VALIDATION_TFRECORDS)
