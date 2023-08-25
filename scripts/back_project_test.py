import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import argparse
import cv2
import numpy as np

from util.queue import load_queue
from util.processing import get_bg_pixels, get_avg_rgb, bg_to_flake_color, mask_flake_color


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

    # TODO: don't hard code this
    # model = cv2.imread(f"C:\\04_03_23_EC_1\\Scan 002\\TileScan_001\\TileScan_001--Stage1397.jpg")
    model = cv2.imread(f"D:\\Graphene\\04_11_23_EC_1\\04_11_23_EC_1\\Scan 001\\TileScan_001\\TileScan_001--Stage672.jpg")
    model = cv2.cvtColor(model, cv2.COLOR_BGR2RGB)

    # Filter out dark, non-flake chunks that will stay dark after equalization
    pixout = get_bg_pixels(model)
    back_rgb = get_avg_rgb(pixout)
    flake_avg_rgb = bg_to_flake_color(back_rgb)
    flake_avg_hsv = cv2.cvtColor(np.uint8([[flake_avg_rgb]]), cv2.COLOR_RGB2HSV)[0][0]  # TODO: hacky?

    mask = mask_flake_color(model, flake_avg_hsv)
    model_hist = cv2.calcHist([model], [0], mask, [180], [0, 180])

    for s in args.s:
        img = cv2.imread(f"{input_dir}\\TileScan_001--Stage{str(s).zfill(3)}.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        start = time.time()
        dst = cv2.calcBackProject([img], [0], model_hist, [0, 180], 1)
        end = time.time()

        print(f"Finished back projection for {s} in {end - start} seconds")

        name = str(time.time_ns())
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey()

        cv2.imshow(name, dst)
        cv2.waitKey()
