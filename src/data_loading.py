import tensorflow as tf

from config import IMG_SIZE


def _parse_function(proto):
    f = {
        "x": tf.FixedLenSequenceFeature([IMG_SIZE[0] * IMG_SIZE[1]], tf.float32, default_value=0., allow_missing=True),
        "label": tf.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True)
    }
    parsed_features = tf.parse_single_example(proto, f)

    x = tf.reshape(parsed_features['x'] / 255, (IMG_SIZE[0], IMG_SIZE[1], 1))
    y = tf.cast(parsed_features['label'], tf.float32)
    return x, y


def load_dataset(input_path, batch_size, shuffle_buffer):
    """
    Create an iterator over the TFRecords file with chroma features.

    :param input_path: the path to the tfrecords file
    :param batch_size:
    :param shuffle_buffer:
    :return: dataset.make_one_shot_iterator()
    """
    dataset = tf.data.TFRecordDataset(input_path)
    dataset = dataset.shuffle(shuffle_buffer).repeat()  # shuffle and repeat
    dataset = dataset.map(_parse_function, num_parallel_calls=16)
    dataset = dataset.batch(batch_size).prefetch(1)  # batch and prefetch

    return dataset
