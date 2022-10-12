from PIL import Image
import numpy as np
import skimage
import matplotlib
import matplotlib.pyplot as plt   # DEBUG

# https://towardsdatascience.com/image-processing-with-python-5b35320a4f3c
# execfile('prepare_images.py')


def downsample_image():
    # Convert image to array
    img = Image.open('data/images/test.jpg')
    img.load()
    img_array = np.asarray(img, dtype='int32')

    # Downsample image
    ds_factor = 10
    ds_array = img_array/255
    r = skimage.measure.block_reduce(ds_array[:, :, 0], (ds_factor, ds_factor), np.mean)
    g = skimage.measure.block_reduce(ds_array[:, :, 1], (ds_factor, ds_factor), np.mean)
    b = skimage.measure.block_reduce(ds_array[:, :, 2], (ds_factor, ds_factor), np.mean)
    ds_array = np.stack((r, g, b), axis=-1)

    # DEBUG
    #print(ds_array.shape)
    # plt.imshow(ds_array)
    # plt.show()

    # Save image
    plt.imsave('./data/ds_images/test_ds.jpg', ds_array)


def main():
    downsample_image()


if __name__ == '__main__':
    main()
