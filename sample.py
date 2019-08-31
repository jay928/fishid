from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os

import matplotlib.pylab as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_hub import registry
import numpy as np
import PIL.Image as Image
from ilsvrc_target_ids import target_ids
import shutil

def train(args):
    pass

def main():
    print('initializing..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='data/pre')
    parser.add_argument('--output_dir', default='data/refined/1st')
    parser.add_argument('--classifier', default='resnet-152')
    args = parser.parse_args()


    classifier_url = "https://tfhub.dev/google/imagenet/resnet_v2_152/classification/3"
    # module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_152/classification/3")
    # height, width = hub.get_expected_image_size(module)
    module = hub.Module(classifier_url, tags=[])
    classifier = tf.keras.Sequential([
        hub.KerasLayer(module)
    ])

    imagenet_labels = np.array(open('imagenet_labels.txt').read().splitlines())

    height, width = hub.get_expected_image_size(module)
    image_shape = (width, height)
    print(image_shape)

    output_dir_p = os.path.join(args.output_dir, 'p')
    output_dir_n = os.path.join(args.output_dir, 'n')


    for dir in os.listdir(args.input_dir):
        output_path_list = []
        print(dir)
        dir_path = os.path.join(args.input_dir, dir)
        if os.path.isdir(dir_path):
            image_batch = np.empty((0, height, width, 3))
            print(image_batch)
            for file in os.listdir(dir_path):
                if os.path.splitext(file)[1].lower() not in ('.jpeg', '.jpg', '.png', '.gif'):
                    print(file)
                    continue
                try:
                    img = Image.open(os.path.join(dir_path, file))
                except:
                    print('Exception occurred during opening this file.', file)
                    continue

                if img.mode != 'RGB':
                    print(img.mode)
                    continue

                img = np.array(img)
                print(np.mean(img))
                img = img / 255.0
                print(np.mean(img), img.shape)
                # print(img.shape)

                output_path_list.append(os.path.join(dir, file))

                image_batch = np.concatenate((image_batch, img[np.newaxis, ...].astype(np.float32)), axis=0)

            print('image_batch shape =', image_batch.shape)

            result = classifier.predict(image_batch)
            # print(result.shape)

            predicted_class = np.argmax(result, axis=-1)
            print(predicted_class)

            predicted_class_name = imagenet_labels[predicted_class]
            print(predicted_class_name)

            for path, cls, cls_name in zip(output_path_list, predicted_class, predicted_class_name):
                fname, ext = os.path.splitext(path)
                file_path = (fname[:100] if len(fname) > 100 else fname) + '__' + cls_name + ext
                if (cls - 1) in target_ids:
                    output_path = os.path.join(args.output_dir, 'p', file_path)
                else:
                    output_path = os.path.join(args.output_dir, 'n', file_path)

                os.makedirs(os.path.split(output_path)[0], exist_ok=True)
                shutil.copyfile(os.path.join(args.input_dir, path), output_path)



if __name__ == '__main__':
    main()
