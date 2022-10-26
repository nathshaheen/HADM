import os
from PIL import Image
import numpy as np
import skimage
import matplotlib.pyplot as plt
import argparse

# https://towardsdatascience.com/image-processing-with-python-5b35320a4f3c
# execfile('downsample.py')


def downsample(image_dir):
    # Downsample each image in the directory
    for i in os.listdir(image_dir):
        if i.endswith('.jpg'):
            # Convert image to array
            img = Image.open(image_dir + '/' + i)
            img.load()
            img_array = np.asarray(img, dtype='int32')

            # Downsample image
            ds_factor = 10  # The factor to downsample by
            ds_array = img_array / 255
            r = skimage.measure.block_reduce(ds_array[:, :, 0], (ds_factor, ds_factor), np.mean)
            g = skimage.measure.block_reduce(ds_array[:, :, 1], (ds_factor, ds_factor), np.mean)
            b = skimage.measure.block_reduce(ds_array[:, :, 2], (ds_factor, ds_factor), np.mean)
            ds_array = np.stack((r, g, b), axis=-1)

            # Save image
            plt.imsave(image_dir + '/ds_images/' + i, ds_array)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir',
                        type=str,
                        default='data/images/gazemap_images',
                        help='The directory that contains images to downsample')
    args = parser.parse_args()

    downsample(args.image_dir)
