"""
Note: Currently only configured for Exfoliator tilescans. Very unlikely to work well on other datasets.
"""
import argparse
import glob
import os
import time
from multiprocessing import Pool

import cv2
import numpy as np

from config import threadsave, boundflag, UM_TO_PX, FLAKE_MIN_AREA_UM2, k, FONT, EQUALIZE_OPEN_MORPH_SIZE, \
                   EQUALIZE_OPEN_MORPH_SHAPE, EQUALIZE_CLOSE_MORPH_SIZE, EQUALIZE_CLOSE_MORPH_SHAPE
from util.queue import load_queue
from util.leica import dim_get, pos_get, get_stage
from util.plot import make_plot, location
from util.processing import mask_equalized, mask_outer, apply_morph_open, apply_morph_close, get_lines, \
                            is_edge_image, get_avg_rgb, get_bg_pixels
from util.box import merge_boxes, make_boxes, draw_box, draw_line_angles
from util.logger import logger


def run_file(img_filepath, output_dir, scan_pos_dict, dims):
    tik = time.time()

    try:
        stage = get_stage(img_filepath)

        img = cv2.imread(img_filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_h, img_w, _ = img.shape

        # If there are too many dark pixels in the image, the image is likely at the edge of the scan; return early
        start = time.time()

        if is_edge_image(img):
            return logger.info(f"{img_filepath} - rejected for dark pixels in {time.time() - tik} seconds")

        end = time.time()
        logger.debug(f"Stage{stage} tested for dark pixels in {end - start} seconds")

        # Mask image using thresholds and apply morph operations to reduce false positives
        start = time.time()

        pixout = get_bg_pixels(img)
        back_rgb = get_avg_rgb(pixout)
        back_hsv = cv2.cvtColor(np.uint8([[back_rgb]]), cv2.COLOR_RGB2HSV)[0][0]  # TODO: hacky?

        outer_mask = mask_outer(img_hsv, back_hsv)

        equalized = cv2.equalizeHist(img_gray)
        equalize_mask = mask_equalized(equalized)

        masked = cv2.bitwise_and(outer_mask, equalize_mask)
        dst = apply_morph_close(masked, size=EQUALIZE_CLOSE_MORPH_SIZE, shape=EQUALIZE_CLOSE_MORPH_SHAPE)
        dst = apply_morph_open(dst, size=EQUALIZE_OPEN_MORPH_SIZE, shape=EQUALIZE_OPEN_MORPH_SHAPE)

        end = time.time()
        logger.debug(f"Stage{stage} thresholded and transformed in {end - start} seconds")

        # Find contours of masked and processed image
        start = time.time()
        contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        end = time.time()

        if len(contours) < 1:
            return logger.info(f"{img_filepath} - rejected for no contours in {time.time() - tik} seconds")
        logger.debug(f"Stage{stage} had {len(contours)} contours in {end - start} seconds")

        # Make boxes and merge boxes that overlap
        start = time.time()

        boxes = make_boxes(contours, hierarchy, img_h, img_w)
        boxes = merge_boxes(boxes)
        boxes = merge_boxes(boxes)

        end = time.time()
        logger.debug(f"Stage{stage} generated and merged boxes in {end - start} seconds")

        if not boxes:
            return logger.info(f"{img_filepath} - rejected for no boxes in {time.time() - tik} seconds")

        xd, yd = location(stage, dims)

        # Convert back from (x, y) scan number to mm coordinates
        try:
            posy, posx = scan_pos_dict[int(yd), int(xd)]
            pos_str = "X:" + str(round(1000 * posx, 2)) + ", Y:" + str(round(1000 * posy, 2))
        except IndexError:
            logger.warn(f'Stage{stage} pos conversion failed!')
            pos_str = ""

        # Label output images
        start = time.time()

        img0 = cv2.putText(img, pos_str, (100, 100), FONT, 3, (0, 0, 0), 2, cv2.LINE_AA)
        img4 = img0.copy()

        max_area = 0

        with open(output_dir + "Color Log.csv", "a+") as flake_log, \
             open(output_dir + "Edge Log.csv", "a+") as edge_log:
            for box in boxes:
                img0 = draw_box(img0, box)
                max_area = max(int(box.area), max_area)

                if boundflag:
                    logger.debug('Drawing contour bounds...')
                    img4 = draw_box(img4, box)
                    img4 = cv2.drawContours(img4, box.contours, -1, (255, 255, 255), 1)

                    lines = get_lines(img4, box.contours)
                    angles = draw_line_angles(img4, box, lines)

                    if lines is not None:
                        edge_log.write(f'{str(stage)},{" ".join(map(str, angles)) if len(angles) > 0 else "-"}')

                flake_log.write(f'{str(stage)},{str(box.area)}\n')

        end = time.time()

        logger.debug(f"Stage{stage} labelled images in {end - start} seconds")

        start = time.time()
        cv2.imwrite(os.path.join(output_dir, os.path.basename(img_filepath)), cv2.cvtColor(img0, cv2.COLOR_RGB2BGR))

        if boundflag:
            cv2.imwrite(os.path.join(output_dir + "\\AreaSort\\", str(max_area) + '_' + os.path.basename(img_filepath)), cv2.cvtColor(img4, cv2.COLOR_RGB2BGR))

        end = time.time()
        logger.debug(f"Stage{stage} saved images in {end - start} seconds")

    except Exception as e:
        logger.warn(f"Exception occurred: {e}")

    tok = time.time()
    logger.info(f"{img_filepath} - {tok - tik} seconds")


def main(args):
    config = load_queue(args.q)

    for input_dir, output_dir in config:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir + "\\AreaSort\\", exist_ok=True)

        input_files = [f for f in glob.glob(os.path.join(input_dir, "*")) if "Stage" in f]
        input_files.sort(key=len)

        # Write log file headers
        with open(output_dir + "Color Log.csv", "w+") as flake_log, \
             open(output_dir + "Edge Log.csv", "w+") as edge_log:
            flake_log.write('N,A\n')
            edge_log.write('N,T\n')

        tik = time.time()
        scanposdict = pos_get(input_dir)
        dims = dim_get(input_dir)

        n_proc = os.cpu_count() - threadsave
        files = [
            [f, output_dir, scanposdict, dims] for f in input_files
            if os.path.splitext(f)[1] in [".jpg", ".png", ".jpeg"]
        ]

        with Pool(n_proc) as pool:
            pool.starmap(run_file, files)

        tok = time.time()

        output_files = [
            f for f in glob.glob(os.path.join(output_dir, "*"))
            if os.path.splitext(f)[1] in [".jpg", ".png", ".jpeg"] and "Stage" in f
        ]
        filecount = len(output_files)

        with open(output_dir + "Summary.txt", "a+") as f:
            f.write(f"Total for {len(files)} files: {tok - tik} = avg of {(tok - tik) / len(files)} per file on {n_proc} logical processors\n")
            f.write(str(filecount) + ' identified flakes\n')

            f.write('t_min_cluster_pixel_count=' + str(FLAKE_MIN_AREA_UM2 * (UM_TO_PX ** 2)) + '\n')
            f.write('k=' + str(k) + "\n\n")

        area_log = open(output_dir + "By Area.csv", "w+")
        area_log.write("Num,A\n")
        area_log.close()
        area_log = open(output_dir + "By Area.csv", "a+")

        start = time.time()
        stages = np.sort(np.array([get_stage(file) for file in output_files]))
        make_plot(stages, dims, output_dir)  # creating cartoon for file
        end = time.time()

        logger.info(f"Created coordmap.jpg in {end - start} seconds")

        flake_data = np.loadtxt(output_dir + "Color Log.csv", skiprows=1, delimiter=',', unpack=True)
        if flake_data.size > 0:
            N, A = flake_data

            pairs = []
            i = 0
            while i < len(A):
                pair = np.array([N[i], A[i]])
                pairs.append(pair)
                i = i + 1

            pairsort = sorted(pairs, key=lambda x: x[1], reverse=True)
            for pair in pairsort:
                writestr = str(int(pair[0])) + ', ' + str(pair[1]) + '\n'
                area_log.write(writestr)

        area_log.close()

        logger.info(f"Total for {len(files)} files: {tok - tik} = avg of {(tok - tik) / len(files)} per file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find graphene flakes on SiO2. Currently configured only for exfoliator dataset"
    )
    parser.add_argument(
        "--q",
        type=str,
        default="Queue.txt",
        help="Directory containing images to process. Optional unless running in headless mode"
    )
    args = parser.parse_args()
    main(args)
