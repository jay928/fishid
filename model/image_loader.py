from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import tensorflow as tf
import tensorflow_hub as hub
from keras.utils import np_utils
import numpy as np

# tf.enable_eager_execution()
# PATH = '/Volumes/SD/deeplearning/data/fish'

PATH = '/data/fishid/source/'
TARGET_SIZE = 224
COUNT = 25
NOISED = [16, 17, 18, 19]

x_data = []
y_data = []
for i in range(0, COUNT):
    print(i)
    if i in NOISED:
        continue

    x_part = np.load(PATH + "x" + str(i) + "_" + str(TARGET_SIZE) + '.npy', allow_pickle=True)
    y_part = np.load(PATH + "y" + str(i) + "_" + str(TARGET_SIZE) + '.npy', allow_pickle=True)
    #     x_part = tf.image.convert_image_dtype(x_part, tf.float32)

    if i == 0:
        x_data = x_part
        y_data = y_part
        continue

    x_data = np.concatenate((x_data, x_part), axis=0)
    y = y_data[-1] + 1
    for j in range(0, len(x_part)):
        y_data = np.concatenate((y_data, [y]), axis=0)

#     y_data = np.concatenate((y_data, y_part), axis=0)

x_data = x_data / 255.0
y_data = np_utils.to_categorical(y_data)
# labels = np.load(PATH + "l_" + str(TARGET_SIZE) + '.npy', allow_pickle=True)[:COUNT]
# labels = [l for idx, l in enumerate(labels) if idx not in NOISED]