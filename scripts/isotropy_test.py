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

        # Benchmark anisotropic image segmentation
        # https://docs.opencv.org/3.4/d4/d70/tutorial_anisotropic_image_segmentation_by_a_gst.html
        start = time.time()
        img_f32 = img_gray.astype(np.float32)

        # J = (J11 J12; J12 J22) - GST
        sobel_x = cv2.Sobel(img_f32, cv2.CV_32F, 1, 0, 3)
        sobel_y = cv2.Sobel(img_f32, cv2.CV_32F, 0, 1, 3)
        sobel_xy = cv2.multiply(sobel_x, sobel_y)

        sobel_xx = cv2.multiply(sobel_x, sobel_x)
        sobel_yy = cv2.multiply(sobel_y, sobel_y)

        J11 = cv2.boxFilter(sobel_xx, cv2.CV_32F, (52, 52))
        J22 = cv2.boxFilter(sobel_yy, cv2.CV_32F, (52, 52))
        J12 = cv2.boxFilter(sobel_xy, cv2.CV_32F, (52, 52))

        # lambda1 = 0.5*(J11 + J22 + sqrt((J11-J22)^2 + 4*J12^2))
        # lambda2 = 0.5*(J11 + J22 - sqrt((J11-J22)^2 + 4*J12^2))
        tmp1 = J11 + J22
        tmp2 = J11 - J22
        tmp4 = np.sqrt(cv2.multiply(tmp2, tmp2) + 4.0 * cv2.multiply(J12, J12))
        lambda1 = 0.5 * (tmp1 + tmp4)  # biggest eigenvalue
        lambda2 = 0.5 * (tmp1 - tmp4)  # smallest eigenvalue

        # Coherency = (lambda1 - lambda2)/(lambda1 + lambda2)) - measure of anisotropism
        # Coherency is anisotropy degree (consistency of local orientation)
        img_coherency = cv2.divide(lambda1 - lambda2, lambda1 + lambda2)

        # tan(2*Alpha) = 2*J12/(J22 - J11)
        # Alpha = 0.5 atan2(2*J12/(J22 - J11))
        img_orientation = cv2.phase(J22 - J11, 2.0 * J12, angleInDegrees=True)
        img_orientation = 0.5 * img_orientation

        _, coherency_mask = cv2.threshold(img_coherency, 0.43, 255, cv2.THRESH_BINARY)
        _, orientation_mask = cv2.threshold(img_orientation, 35, 57, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(coherency_mask, orientation_mask)

        end = time.time()
        print(f"Finished anisotropic segmentation for {s} in {end - start} seconds")

        name = str(time.time_ns())
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, img)
        cv2.waitKey()

        cv2.imshow(name, img_coherency)
        cv2.waitKey()

        cv2.imshow(name, coherency_mask)
        cv2.waitKey()

        cv2.imshow(name, img_orientation)
        cv2.waitKey()

        cv2.imshow(name, orientation_mask)
        cv2.waitKey()

        cv2.imshow(name, mask)
        cv2.waitKey()
