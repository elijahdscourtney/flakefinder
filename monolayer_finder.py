"""
Note: Currently only configured for Exfoliator tilescans. Very unlikely to work well on other datasets.
"""
import argparse
import glob
import os
import re
import time
from multiprocessing import Pool

import cv2
import numpy as np
import matplotlib
from sklearn.cluster import DBSCAN

from util.config import load_config
from util.leica import dim_get, pos_get
from util.plot import make_plot, location
from util.processing import bg_to_flake_color, edgefind, merge_boxes, Box
from util.logger import logger


flake_colors_rgb = [
    # # Thick-looking
    # [6, 55, 94],
    # Monolayer-looking
    # [57, 65, 86],
    # [60, 66, 85],
    # [89,99,109],
    [0, 0, 0],
]
flake_colors_hsv = [
    np.uint8(matplotlib.colors.rgb_to_hsv(x) * np.array([179, 255, 255])) for x in flake_colors_rgb
    # matplotlib outputs in range 0,1. Opencv expects HSV images in range [0,0,0] to [179, 255,255]
]
flake_color_hsv = np.mean(flake_colors_hsv, axis=0)
avg_rgb = np.mean(flake_colors_rgb, axis=0)

output_dir = 'cv_output'
threadsave = 1  # number of threads NOT allocated when running
boundflag = 1
t_rgb_dist = 8
# t_hue_dist = 12 #12
t_red_dist = 12
# t_red_cutoff = 0.1 #fraction of the chunked image that must be more blue than red to be binned
t_color_match_count = 0.000225  # fraction of image that must look like monolayers
k = 4
t_min_cluster_pixel_count = 30 * (k / 4) ** 2  # flake too small
t_max_cluster_pixel_count = 20000 * (k / 4) ** 2  # flake too large
# scale factor for DB scan. recommended values are 3 or 4. Trade-off in time vs accuracy. Impact epsilon.
scale = 1  # the resolution images are saved at, relative to the original file. Does not affect DB scan


# This would be a decorator but apparently multiprocessing lib doesn't know how to serialize it.
def run_file_wrapped(filepath):
    tik = time.time()
    filepath1 = filepath[0]
    outputloc = filepath[1]
    scanposdict = filepath[2]
    dims = filepath[3]
    try:
        run_file(filepath1, outputloc, scanposdict, dims)
    except Exception as e:
        logger.warn(f"Exception occurred: {e}")
    tok = time.time()
    logger.info(f"{filepath[0]} - {tok - tik} seconds")


def run_file(img_filepath, outputdir, scanposdict, dims):
    tik = time.time()
    img0 = cv2.imread(img_filepath)
    img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape
    pixcal = 1314.08 / w  # microns/pixel from Leica calibration
    pixcals = [pixcal, 876.13 / h]
    img_pixels = img.copy().reshape(-1, 3)
    lowlim = np.array([87, 100, 99])  # np.array([108,100,99])#defines lower limit for what code can see as background
    highlim = np.array([114, 118, 114])  # np.array([140,165,135])
    imsmall = cv2.resize(img.copy(), dsize=(256 * k, 171 * k)).reshape(-1, 3)
    test = np.sign(imsmall - lowlim) + np.sign(highlim - imsmall)
    pixout = imsmall * np.sign(
        test + abs(test))  # chooses pixels between provided limits, quickly filtering to potential background pixels
    if len(pixout) == 0:  # making sure background is identified
        # print('Pixel failed')
        return
    reds = np.bincount(pixout[:, 0])
    greens = np.bincount(pixout[:, 1])
    blues = np.bincount(pixout[:, 2])
    reds[0] = 0  # otherwise argmax finds values masked to 0 by pixout
    greens[0] = 0
    blues[0] = 0
    # print(reds,len(reds))
    reddest = reds.argmax()
    greenest = greens.argmax()
    bluest = blues.argmax()
    backrgb = [reddest, greenest, bluest]  # defining background color
    # print(backrgb)
    avg_rgb = bg_to_flake_color(backrgb)  # calculates monolayer color based on background color
    # print(backrgb,avg_rgb)
    rgb_pixel_dists = np.sqrt(
        np.sum((img_pixels - avg_rgb) ** 2, axis=1))  # calculating distance between each pixel and predicted flake RGB
    # rgb_t_count = np.sum(rgb_pixel_dists < t_rgb_dist)

    # hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # hsv_pixels = hsv_img.reshape(-1, 3)
    # hue_pixel_dists = np.sqrt((hsv_pixels[:, 0] - flake_color_hsv[0]) ** 2)
    # hue_t_count = np.sum(hue_pixel_dists < t_hue_dist)

    # masking the image to only the pixels close enough to predicted flake color
    # img_mask = np.logical_and(hue_pixel_dists < t_hue_dist, rgb_pixel_dists < t_rgb_dist)
    img_mask = np.logical_and(rgb_pixel_dists < t_rgb_dist, reddest - img_pixels[:, 0] > 5)

    # Show how many are true, how many are false.
    t_count = np.sum(img_mask)
    # print(t_count)
    if t_count < t_color_match_count * len(img_pixels):
        # print('Count failed',t_count)
        return
    logger.debug(f"{img_filepath} meets count thresh with {t_count}")
    pixdark = np.sum((img_pixels[:, 2] < 25) * (img_pixels[:, 1] < 25) * (img_pixels[:, 0] < 25))
    if np.sum(pixdark) / len(img_pixels) > 0.1:  # edge detection, if more than 10% of the image is too dark, return
        logger.debug(f"{img_filepath} was on an edge!")
        return
    # Create Masked image
    img2_mask_in = img.copy().reshape(-1, 3)
    img2_mask_in[~img_mask] = np.array([0, 0, 0])
    img2_mask_in = img2_mask_in.reshape(img.shape)

    # DB SCAN, fitting to find clusters of correctly colored pixels
    dbscan_img = cv2.cvtColor(img2_mask_in, cv2.COLOR_RGB2GRAY)
    dbscan_img = cv2.resize(dbscan_img, dsize=(256 * k, 171 * k))
    # db = DBSCAN(eps=2.0, min_samples=6, metric='euclidean', algorithm='auto', n_jobs=-1)
    db = DBSCAN(eps=2.0, min_samples=6, metric='euclidean', algorithm='auto', n_jobs=1)
    indices = np.dstack(np.indices(dbscan_img.shape[:2]))
    xycolors = np.concatenate((np.expand_dims(dbscan_img, axis=-1), indices), axis=-1)
    feature_image = np.reshape(xycolors, [-1, 3])
    db.fit(feature_image)
    label_names = range(-1, db.labels_.max() + 1)
    # print(f"{img_filepath} had {len(label_names)}  dbscan clusters")

    # Thresholding of clusters
    labels = db.labels_
    n_pixels = np.bincount(labels + 1, minlength=len(label_names))
    # print(n_pixels)
    criteria = np.logical_and(n_pixels > t_min_cluster_pixel_count, n_pixels < t_max_cluster_pixel_count)
    h_labels = np.array(label_names)
    # print(h_labels)
    h_labels = h_labels[criteria]
    # print(h_labels)
    h_labels = h_labels[h_labels > 0]

    if len(h_labels) < 1:
        return
    logger.debug(f"{img_filepath} had {len(h_labels)} filtered dbscan clusters")

    # Make boxes
    boxes = []
    for label_id in h_labels:
        # Find bounding box... in x/y plane find min value. This is just argmin and argmax
        criteria = labels == label_id
        criteria = criteria.reshape(dbscan_img.shape[:2]).astype(np.uint8)
        x = np.where(criteria.sum(axis=0) > 0)[0]
        y = np.where(criteria.sum(axis=1) > 0)[0]
        width = x.max() - x.min()
        height = y.max() - y.min()
        boxes.append(Box(label_id, x.min(), y.min(), width, height))

    # Merge boxes
    pass_1 = merge_boxes(dbscan_img, boxes)
    merged_boxes = merge_boxes(dbscan_img, pass_1)

    if not merged_boxes:
        return

    # Make patches out of clusters
    wantshape = (int(int(img.shape[1]) * scale), int(int(img.shape[0]) * scale))
    bscale = wantshape[0] / (256 * k)  # need to scale up box from dbscan image
    offset = 5
    patches = [
        [int((int(b.x) - offset) * bscale), int((int(b.y) - offset) * bscale),
         int((int(b.width) + 2 * offset) * bscale), int((int(b.height) + 2 * offset) * bscale)] for b
        in merged_boxes
    ]
    logger.debug('patched')
    color = (0, 0, 255)
    thickness = 6
    log_file = open(outputdir + "Color Log.txt", "a+")

    stage = int(re.search(r"Stage(\d{3})", img_filepath).group(1))
    imloc = location(stage, dims)

    radius = 1
    i = -1
    while radius > 0.1:
        i = i + 1
        radius = (int(imloc[0]) - int(scanposdict[i][0])) ** 2 + (int(imloc[1]) - int(scanposdict[i][2])) ** 2
    posx = scanposdict[i][1]
    posy = scanposdict[i][3]
    posstr = "X:" + str(round(1000 * posx, 2)) + ", Y:" + str(round(1000 * posy, 2))

    img0 = cv2.putText(img0, posstr, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2, cv2.LINE_AA)
    img4 = img0.copy()

    for p in patches:
        logger.debug(p)
        y_min = int(p[0] + 2 * offset * bscale / 3)
        y_max = int(p[0] + p[2] - 2 * offset * bscale / 3)
        x_min = int(p[1] + 2 * offset * bscale / 3)
        x_max = int(p[1] + p[3] - 2 * offset * bscale / 3)  # note that the offsets cancel here
        logger.debug((x_min, y_min, x_max, y_max))
        bounds = [max(0, p[1]), min(p[1] + p[3], int(h)), max(0, p[0]), min(p[0] + p[2], int(w))]
        imchunk = img[bounds[0]:bounds[1], bounds[2]:bounds[3]]  # identifying bounding box of flake
        xarr = []
        yarr = []
        width = round(p[2] * pixcal, 1)
        height = round(p[3] * pixcal, 1)  # microns
        img3 = cv2.rectangle(img0, (p[0], p[1]), (p[0] + p[2], p[1] + p[3]), color,
                             thickness)  # creating the output images
        img3 = cv2.putText(img3, str(height), (p[0] + p[2] + 10, p[1] + int(p[3] / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                           (0, 0, 0), 2, cv2.LINE_AA)
        img3 = cv2.putText(img3, str(width), (p[0], p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        flakergb = [0, 0, 0]
        farea = 0

        if boundflag == 1:
            flakergb, indices, farea = edgefind(imchunk, avg_rgb, pixcals, t_rgb_dist)  # calculating border pixels
            logger.debug('Edge found')
            for index in indices:
                # print(index)
                indx = index[0] + bounds[0]
                indy = index[1] + bounds[2]
                img4 = cv2.rectangle(img4, (p[0], p[1]), (p[0] + p[2], p[1] + p[3]), color, thickness)
                img4 = cv2.putText(img4, str(height), (p[0] + p[2] + 10, p[1] + int(p[3] / 2)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                img4 = cv2.putText(img4, str(width), (p[0], p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                                   cv2.LINE_AA)
                img4[indx, indy] = img4[indx, indy] + [25, 25, 25]
                xarr.append(indx)
                yarr.append(indy)
        logstr = str(stage) + ',' + str(farea) + ',' + str(flakergb[0]) + ',' + str(flakergb[1]) + ',' + str(
            flakergb[2]) + ',' + str(backrgb[0]) + ',' + str(backrgb[1]) + ',' + str(backrgb[2])
        log_file.write(logstr + '\n')

    log_file.close()

    cv2.imwrite(os.path.join(outputdir, os.path.basename(img_filepath)), img3)
    if boundflag == 1:
        cv2.imwrite(os.path.join(outputdir + "\\AreaSort\\", str(int(farea)) + '_' + os.path.basename(img_filepath)),
                    img4)

    tok = time.time()
    logger.info(f"{img_filepath} - {tok - tik} seconds")


def main(args):
    config = load_config(args.q)

    for input_dir, output_dir in config:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir + "\\AreaSort\\", exist_ok=True)
        files = glob.glob(os.path.join(input_dir, "*"))

        files = [f for f in files if "Stage" in f]
        files.sort(key=len)
        # Filter files to only have images.
        # smuggling output_dir into pool.map by packaging it with the iterable, gets unpacked by run_file_wrapped
        dims = dim_get(input_dir)

        with open(output_dir + "Color Log.txt", "w+") as logFile:
            logFile.write('N,A,Rf,Gf,Bf,Rw,Gw,Bw\n')

        tik = time.time()
        scanposdict = pos_get(input_dir)
        files = [
            [f, output_dir, scanposdict, dims] for f in files if os.path.splitext(f)[1] in [".jpg", ".png", ".jpeg"]
        ]

        n_proc = os.cpu_count() - threadsave  # config.jobs if config.jobs > 0 else
        with Pool(n_proc) as pool:
            pool.map(run_file_wrapped, files)
        tok = time.time()

        filecounter = glob.glob(os.path.join(output_dir, "*"))
        filecounter = [f for f in filecounter if os.path.splitext(f)[1] in [".jpg", ".png", ".jpeg"]]
        filecounter2 = [f for f in filecounter if "Stage" in f]
        # print(filecounter2)
        # print(filecounter2)

        filecount = len(filecounter2)
        with open(output_dir + "Summary.txt", "a+") as f:
            f.write(f"Total for {len(files)} files: {tok - tik} = avg of {(tok - tik) / len(files)} per file on {n_proc} logical processors\n")
            f.write(str(filecount) + ' identified flakes\n')

            f.write('flake_colors_rgb=' + str(flake_colors_rgb) + '\n')
            f.write('t_rgb_dist=' + str(t_rgb_dist) + '\n')
            # f.write('t_hue_dist='+str(t_hue_dist)+'\n')
            f.write('t_red_dist=' + str(t_red_dist) + '\n')
            # f.write('t_red_cutoff='+str(t_red_cutoff)+'\n')
            f.write('t_color_match_count=' + str(t_color_match_count) + '\n')
            f.write('t_min_cluster_pixel_count=' + str(t_min_cluster_pixel_count) + '\n')
            f.write('t_max_cluster_pixel_count=' + str(t_max_cluster_pixel_count) + '\n')
            f.write('k=' + str(k) + "\n\n")

        flist = open(output_dir + "Imlist.txt", "w+")
        flist.write("List of Stage Numbers for copying to Analysis Sheet" + "\n")
        flist.close()
        flist = open(output_dir + "Imlist.txt", "a+")
        fwrite = open(output_dir + "By Area.txt", "w+")
        fwrite.write("Num, A" + "\n")
        fwrite.close()
        fwrite = open(output_dir + "By Area.txt", "a+")

        numlist = []
        for file in filecounter2:
            splits = file.split("Stage")
            num = splits[1]
            number = os.path.splitext(num)[0]
            numlist.append(int(number))

        numlist = np.sort(np.array(numlist))
        for number in numlist:
            flist.write(str(number) + "\n")

        make_plot(numlist, dims, output_dir)  # creating cartoon for file
        flist.close()

        # print(output_dir+"Color Log.txt")
        N, A, Rf, Gf, Bf, Rw, Gw, Bw = np.loadtxt(output_dir + "Color Log.txt", skiprows=1, delimiter=',', unpack=True)

        pairs = []
        i = 0
        while i < len(A):
            pair = np.array([N[i], A[i]])
            pairs.append(pair)
            i = i + 1
        # print(pairs)
        pairsort = sorted(pairs, key=lambda x: x[1], reverse=True)
        # print(pairs,pairsort)
        for pair in pairsort:
            writestr = str(int(pair[0])) + ', ' + str(pair[1]) + '\n'
            fwrite.write(writestr)
        fwrite.close()

        logger.info(f"Total for {len(files)} files: {tok - tik} = avg of {(tok - tik) / len(files)} per file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find graphene flakes on SiO2. Currently configured only for exfoliator dataset"
    )
    parser.add_argument(
        "--q",
        required=True,
        type=str,
        help="Directory containing images to process. Optional unless running in headless mode"
    )
    args = parser.parse_args()
    main(args)
