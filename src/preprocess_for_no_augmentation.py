from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import glob
import os


PATH_SOURCE = '/Volumes/SD/deeplearning/data/fish/integration'
PATH_NP = '/Volumes/SD/deeplearning/data/fish/np/'

def find_folders(path):
    folders = []

    for r, d, f in os.walk(path):
        for folder in d:
            folders.append(folder)

    return list(set(folders))


def load_and_save(path, labels, count, target_size):
    for i, label in enumerate(labels):
        x_data = []
        y_data = []

        files = [f for f in glob.glob(path + "/" + label + "/*.*", recursive=True)]

        for file in files:
            if i > count:
                break

            print(file)
            try:
                image = load_img(file, target_size=(target_size, target_size))
            except:
                continue

            image_array = img_to_array(image)
            # image_array = image_array.astype('float32')
            # image_array /= 255.0

            x_data.append(image_array)
            y_data.append(labels.index(label))

        np.save(PATH_NP + 'x' + str(i) + "_" + str(TARGET_SIZE), x_data, allow_pickle=True)
        np.save(PATH_NP + 'y' + str(i) + "_" + str(TARGET_SIZE), y_data, allow_pickle=True)

COUNT = 999
TARGET_SIZE = 224
labels = find_folders(PATH_SOURCE)

load_and_save(PATH, labels, COUNT, TARGET_SIZE)

np.save(PATH_NP + "l_" + str(TARGET_SIZE), labels, allow_pickle=True)

print('completed!')