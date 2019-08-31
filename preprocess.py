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
    parser.add_argument('--input_dir', default='data/raw')
    parser.add_argument('--output_dir', default='data/pre2')
    parser.add_argument('--padding', default=False)
    args = parser.parse_args()


    classifier_url = "https://tfhub.dev/google/imagenet/resnet_v2_152/classification/3"
    # module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_152/classification/3")
    # height, width = hub.get_expected_image_size(module)
    module = hub.Module(classifier_url, tags=[])
    classifier = tf.keras.Sequential([
        hub.KerasLayer(module)
    ])

    imagenet_labels = np.array(open('data/imagenet_labels.txt').read().splitlines())

    height, width = hub.get_expected_image_size(module)
    image_shape = (width, height)
    print(image_shape)


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

                img.save('temp1.png')

                img_w, img_h = img.size
                # print(np.mean(img), (img_w, img_h), np.array(img).shape)

                # :param box: The crop rectangle, as a (left, upper, right, lower)-tuple.
                if args.padding:
                    desired_size = max(img_w, img_h)
                    new_im = Image.new("RGB", (desired_size, desired_size))
                    new_im.paste(img, ((desired_size - img_w) // 2, (desired_size - img_h) // 2))
                    img = new_im
                else:
                    if img_w > img_h:
                        diff = (img_w - img_h) // 2
                        img = img.crop((diff, 0, img_w - diff, img_h))
                    else:
                        diff = (img_h - img_w) // 2
                        img = img.crop((0, diff, img_w, img_h - diff))

                img.save('temp2.png')
                img = img.resize(image_shape)
                img = img.resize(image_shape)

                os.makedirs(os.path.join(args.output_dir, dir), exist_ok=True)

                img.save(os.path.join(args.output_dir, dir, file))


if __name__ == '__main__':
    main()