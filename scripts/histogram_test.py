import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import argparse
import cv2
import numpy as np

from util.queue import load_queue
from util.processing import bg_to_flake_color, get_avg_rgb, mask_outer, mask_equalized, get_bg_pixels, RGB

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('tkagg')

use_cv2_hist = True


def make_histogram(img: np.ndarray, mask, title: str, n: int, back: RGB | int, flake: np.ndarray[int] | int, lim: int = 256):
    print(f"Plotting {title}")
    dims = len(img.shape)
    channel = n - 1 if dims == 3 else 0

    if n != -1:
        plt.subplot(3, 1, n)

    if use_cv2_hist:
        plt.plot(cv2.calcHist([img], [channel], mask, [lim], [0, lim]))
    else:
        # TODO: mask
        data = img[:, :, channel].ravel() if dims == 3 else img.ravel()
        plt.hist(data, bins=256, range=(0, 255))

    plt.axvline(back if np.isscalar(back) else back[channel], dashes=(2, 1))
    plt.axvline(flake if np.isscalar(flake) else flake[channel], dashes=(2, 1), color='b')

    plt.ylabel('Occurrences')
    plt.xlim([0, lim])
    plt.title(title)


def show_img(mask=None):
    masked = " (masked)" if mask is not None else ""

    make_histogram(img, mask, f"Stage {s} -- reds" + masked, 1, back_rgb, flake_avg_rgb)
    make_histogram(img, mask, f"Stage {s} -- greens" + masked, 2, back_rgb, flake_avg_rgb)
    make_histogram(img, mask, f"Stage {s} -- blues" + masked, 3, back_rgb, flake_avg_rgb)
    plt.show()

    img2 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    make_histogram(img2, mask, f"Stage {s} -- hues" + masked, 1, back_hsv, flake_avg_hsv, lim=180)
    make_histogram(img2, mask, f"Stage {s} -- saturations" + masked, 2, back_hsv, flake_avg_hsv)
    make_histogram(img2, mask, f"Stage {s} -- values" + masked, 3, back_hsv, flake_avg_hsv)
    plt.show()

    img3 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    make_histogram(img3, mask, f"Stage {s} -- intensities" + masked, -1, back_gray, flake_avg_gray)
    plt.show()


if __name__ == "__main__":
    # TODO: new description or abstract
    parser = argparse.ArgumentParser(
        description="Find graphene flakes on SiO2. Currently configured only for exfoliator dataset"
    )
    parser.add_argument(
        "--s",
        required=True,
        type=int,
        nargs="+",
        help="Scan stages to test (ex. --s 246 248 250)"
    )
    args = parser.parse_args()

    queue = load_queue('Queue.txt')
    input_dir, _ = queue[0]

    for s in args.s:
        img = cv2.imread(f"{input_dir}\\TileScan_001--Stage{str(s).zfill(3)}.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        pixout = get_bg_pixels(img)

        back_rgb = get_avg_rgb(pixout)
        back_hsv = cv2.cvtColor(np.uint8([[back_rgb]]), cv2.COLOR_RGB2HSV)[0][0]  # TODO: hacky?
        back_gray = cv2.cvtColor(np.uint8([[back_rgb]]), cv2.COLOR_RGB2GRAY)[0][0]  # TODO: hacky?

        flake_avg_rgb = bg_to_flake_color(back_rgb)
        flake_avg_hsv = cv2.cvtColor(np.uint8([[flake_avg_rgb]]), cv2.COLOR_RGB2HSV)[0][0]  # TODO: hacky?
        flake_avg_gray = cv2.cvtColor(np.uint8([[flake_avg_rgb]]), cv2.COLOR_RGB2GRAY)[0][0]  # TODO: hacky?

        # Calculate histograms from image
        show_img()

        outer_mask = mask_outer(img_hsv, back_hsv)
        equalize_mask = mask_equalized(cv2.equalizeHist(img_gray))
        show_img(cv2.bitwise_and(outer_mask, equalize_mask))
