import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import argparse
import cv2
import numpy as np

from util.queue import load_queue


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
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Benchmark Sobel derivative gradient detector
        # https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html
        start = time.time()
        grad_x = cv2.Scharr(img_gray, -1, 1, 0)
        grad_x = cv2.convertScaleAbs(grad_x)

        grad_y = cv2.Scharr(img_gray, -1, 0, 1)
        grad_y = cv2.convertScaleAbs(grad_y)

        grad = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)
        end = time.time()

        print(f"Finished Sobel for {s} in {end - start} seconds")

        name = str(time.time_ns())
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, img)
        cv2.waitKey()

        cv2.imshow(name, grad)
        cv2.waitKey()

        # Benchmark Laplace operator gradient detector
        # https://docs.opencv.org/3.4/d5/db5/tutorial_laplace_operator.html
        start = time.time()
        dst = cv2.Laplacian(img_gray, -1, ksize=3)
        dst = cv2.convertScaleAbs(dst)
        end = time.time()

        print(f"Finished Laplace for {s} in {end - start} seconds")

        cv2.imshow(name, img)
        cv2.waitKey()

        cv2.imshow(name, dst)
        cv2.waitKey()
