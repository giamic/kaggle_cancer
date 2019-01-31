import os
from datetime import datetime

# PATH VARIABLES
DATA_FOLDER = os.path.join('..', 'data')
LABELS_PATH = os.path.join(DATA_FOLDER, 'train_labels.csv')

TRAIN_FOLDER = os.path.join(DATA_FOLDER, 'train')
VALIDATION_FOLDER = os.path.join(DATA_FOLDER, 'validation')
TEST_FOLDER = os.path.join(DATA_FOLDER, 'test')

TRAIN_TFRECORDS = os.path.join(DATA_FOLDER, 'train.tfrecords')
VALIDATION_TFRECORDS = os.path.join(DATA_FOLDER, 'validation.tfrecords')
TEST_TFRECORDS = os.path.join(DATA_FOLDER, 'test.tfrecords')

# DATA PARAMETERS
IMG_SIZE = (96, 96)
N_TRAIN = 197979  # comes from our train.tfrecords file
N_VALIDATION = 22046  # comes from our validation.tfrecords file

# MODEL PARAMETERS
