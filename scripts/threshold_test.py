import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import argparse
import cv2
import numpy as np

from util.queue import load_queue
from util.box import make_boxes, merge_boxes, draw_box, draw_line_angles
from util.processing import bg_to_flake_color, get_bg_pixels, get_avg_rgb, mask_flake_color, apply_morph_open, apply_morph_close, get_lines


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
        # Run all the flake color logic first, since that isn't what's being benchmarked here
        img = cv2.imread(f"{input_dir}\\TileScan_001--Stage{str(s).zfill(3)}.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pixout = get_bg_pixels(img)

        back_rgb = get_avg_rgb(pixout)
        flake_avg_rgb = bg_to_flake_color(back_rgb)
        flake_avg_hsv = cv2.cvtColor(np.uint8([[flake_avg_rgb]]), cv2.COLOR_RGB2HSV)[0][0]  # TODO: hacky?

        # Benchmark cv2 thresholding
        tik = time.time()
        masked = mask_flake_color(img, flake_avg_hsv)
        tok = time.time()

        name = str(time.time_ns())
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, masked)
        cv2.waitKey()

        print(f"Finished threshold mask for {s} in {tok - tik} seconds")

        tik = time.time()
        dst = apply_morph_close(masked)
        tok = time.time()

        cv2.imshow(name, dst)
        cv2.waitKey()

        print(f"Finished close morph in {tok - tik} seconds")

        tik = time.time()
        dst = apply_morph_open(dst)
        tok = time.time()

        cv2.imshow(name, dst)
        cv2.waitKey()

        print(f"Finished open morph in {tok - tik} seconds")

        tik = time.time()
        contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        dst = cv2.drawContours(cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR), contours, -1, (0, 0, 255), 2)
        tok = time.time()

        cv2.imshow(name, dst)
        cv2.waitKey()

        print(f"Finished contour search in {tok - tik} seconds")

        tik = time.time()
        boxes = make_boxes(contours, hierarchy, img.shape[0], img.shape[1])
        boxes = merge_boxes(boxes)
        boxes = merge_boxes(boxes)
        tok = time.time()

        print(f"Generated and merged boxes in {tok - tik} seconds")

        for box in boxes:
            img = draw_box(img, box)
            img = cv2.drawContours(img, box.contours, -1, (255, 255, 255), 1)

            lines = get_lines(img, box.contours)
            draw_line_angles(img, box, lines)

        cv2.imshow(name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey()

        # baseline = cv2.imread(f"C:\\04_03_23_EC_1\\MLScanned2\\TileScan_001--Stage{s}.jpg")
        # cv2.imshow(name, baseline)
        # cv2.waitKey()

        print("-----")
