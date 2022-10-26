import os
from PIL import Image
import numpy as np
import argparse


def grid_calculation(image_array, num_grid):
    # Initialize grid to 0
    grid = np.zeros(num_grid[0] * num_grid[1], dtype=float)

    # Filter gaze map pixels < 0.15
    Y, X = np.nonzero(image_array > 0.15)

    image_size = image_array.shape

    x_grid = image_size[1] // num_grid[1]
    y_grid = image_size[0] // num_grid[0]

    # Count the pixels >= 0.15
    for x, y in zip(X, Y):
        idx = y // y_grid * num_grid[1] + x // x_grid
        grid[idx] += 1
    grid = grid / np.sum(grid)

    return grid


def main(args):
    f = open(args.grids, "a")
    count = 0
    for root, dirs, files in os.walk(args.gazemaps):
        for item in files:
            entry = []
            name = item.split('_')
            nn = name[0] + '_' + name[-1]
            entry.append(nn)
            gt = np.array(Image.open(os.path.join(args.gazemaps, item)).convert('L'))

            gt = (gt - np.min(gt)) / ((np.max(gt) - np.min(gt)) * 1.0)  # Normalize

            gt_grid = grid_calculation(gt, [args.gridheight, args.gridwidth])

            entry.extend(gt_grid)

            s = ','.join(map(str, entry))
            f.write(s + '\n')
            count += 1

            # User feedback
            if count % 500 == 0:
                print("Count: %d" % count)
    print(count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create ground-truth grid for gaze maps')
    parser.add_argument('--gazemaps', metavar='DIR', help='path to gaze map images folder')
    parser.add_argument('--grids', metavar='PATH', help='path to output txt file')
    parser.add_argument('--gridheight', default=16, type=int, metavar='N',
                        help='number of rows in grid')
    parser.add_argument('--gridwidth', default=16, type=int, metavar='N',
                        help='number of columns in grid ')
    args = parser.parse_args()

    main(args)
