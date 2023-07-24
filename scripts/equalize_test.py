import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import argparse
import cv2
import numpy as np

from config import EQUALIZE_OPEN_MORPH_SIZE, EQUALIZE_CLOSE_MORPH_SIZE, EQUALIZE_OPEN_MORPH_SHAPE, EQUALIZE_CLOSE_MORPH_SHAPE
from util.queue import load_queue
from util.processing import mask_equalized, mask_dark_pixels, mask_contrast, apply_morph_open, apply_morph_close, get_lines
from util.box import make_boxes, merge_boxes, draw_box, draw_line_angles


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
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        name = str(time.time_ns())
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cv2.waitKey()

        cv2.imshow(name, img_gray)
        cv2.waitKey()

        # Filter out dark, non-flake chunks that will stay dark after equalization
        contrast_mask = mask_contrast(img)
        cv2.imshow(name, contrast_mask)
        cv2.waitKey()

        start = time.time()
        equalized = cv2.equalizeHist(img_gray)
        end = time.time()

        print(f"Equalized histogram for {s} in {end - start} seconds")

        cv2.imshow(name, equalized)
        cv2.waitKey()

        equalize_mask = mask_equalized(equalized)
        cv2.imshow(name, equalize_mask)
        cv2.waitKey()

        masked = cv2.bitwise_and(contrast_mask, equalize_mask)
        cv2.imshow(name, masked)
        cv2.waitKey()

        # Treat threshold like previous `mask_flake_color()` mask and run rest of algorithm on it
        masked = apply_morph_open(masked, size=EQUALIZE_OPEN_MORPH_SIZE, shape=EQUALIZE_OPEN_MORPH_SHAPE)
        cv2.imshow(name, masked)
        cv2.waitKey()

        masked = apply_morph_close(masked, size=EQUALIZE_CLOSE_MORPH_SIZE, shape=EQUALIZE_CLOSE_MORPH_SHAPE)
        cv2.imshow(name, masked)
        cv2.waitKey()

        contours, hierarchy = cv2.findContours(masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        dst = cv2.drawContours(cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR), contours, -1, (0, 0, 255), 2)
        cv2.imshow(name, dst)
        cv2.waitKey()

        boxes = make_boxes(contours, hierarchy, img.shape[0], img.shape[1])
        boxes = merge_boxes(masked, boxes)
        boxes = merge_boxes(masked, boxes)

        for box in boxes:
            img = draw_box(img, box)
            img = cv2.drawContours(img, box.contours, -1, (255, 255, 255), 1)

            lines = get_lines(img, box.contours)
            draw_line_angles(img, box, lines)

        cv2.imshow(name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey()
