import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import cv2
import numpy as np

from util.processing import bg_to_flake_color, get_avg_rgb

k = 4
t_rgb_dist = 8


def classical(img0):
    img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

    img_pixels = img.copy().reshape(-1, 3)
    rgb_pixel_dists = np.sqrt(np.sum((img_pixels - flake_avg_rgb) ** 2, axis=1))

    img_mask = np.logical_and(rgb_pixel_dists < t_rgb_dist, back_rgb[0] - img_pixels[:, 0] > 5)
    # t_count = np.sum(img_mask)

    img2_mask_in = img.copy().reshape(-1, 3)
    img2_mask_in[~img_mask] = np.array([0, 0, 0])

    return img2_mask_in.reshape(img.shape)


def threshold(img0):
    img = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)

    lower = tuple(map(int, flake_avg_hsv - (6, 25, 25)))
    higher = tuple(map(int, flake_avg_hsv + (6, 25, 25)))

    return cv2.inRange(img, lower, higher)


if __name__ == "__main__":
    # Run all the flake color logic first, since that isn't what's being benchmarked here
    # TODO: make this more efficient?
    img0 = cv2.imread("C:\\04_03_23_EC_1\\Scan 002\\TileScan_001\\TileScan_001--Stage246.jpg")
    img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

    lowlim = np.array([87, 100, 99])  # defines lower limit for what code can see as background
    highlim = np.array([114, 118, 114])

    imsmall = cv2.resize(img.copy(), dsize=(256 * k, 171 * k)).reshape(-1, 3)
    test = np.sign(imsmall - lowlim) + np.sign(highlim - imsmall)
    pixout = imsmall * np.sign(test + abs(test))

    back_rgb = get_avg_rgb(pixout)
    flake_avg_rgb = bg_to_flake_color(back_rgb)
    flake_avg_hsv = cv2.cvtColor(np.uint8([[flake_avg_rgb]]), cv2.COLOR_RGB2HSV)[0][0]  # TODO: hacky?

    # Benchmark classical
    tik = time.time()
    masked = classical(img0)
    tok = time.time()

    cv2.namedWindow("classical", cv2.WINDOW_NORMAL)
    cv2.imshow("classical", cv2.cvtColor(masked, cv2.COLOR_RGB2BGR))
    cv2.waitKey()

    print(f"Finished classical in {tok - tik} seconds")

    # Benchmark cv2 thresholding
    tik = time.time()
    masked = threshold(img0)
    tok = time.time()

    cv2.imshow("classical", masked)
    cv2.waitKey()

    print(f"Finished threshold in {tok - tik} seconds")
