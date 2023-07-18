"""
Note: Currently only configured for Exfoliator tilescans. Very unlikely to work well on other datasets.
"""
import argparse
import glob
import os
import time
from multiprocessing import Pool

import cv2
import matplotlib
import numpy as np

from util.queue import load_queue
from util.leica import dim_get

matplotlib.use('tkagg')


threadsave = 4  # number of threads NOT allocated when running
rescale = 1 / 60


# This would be a decorator but apparently multiprocessing lib doesn't know how to serialize it.
def run_file_wrapped(filelist):
    tik = time.time()
    try:
        run_file(filelist)
    except Exception as e:
        print("Exception occurred: ", e)
    tok = time.time()
    # print(f"{filelist[0]} - {tok - tik} seconds")


def run_file(filelist):
    filepath = filelist[0]
    newshape = filepath[2]
    dims = filepath[3]
    outputloc = filepath[4]
    stitch = np.int16(np.zeros(newshape))
    for filepath in filelist:
        img_filepath = filepath[0]
        step = filepath[1]
        # print(filepath[1])
        # print('newshape',img_filepath[2])

        tik = time.time()
        # print(img_filepath)
        img0 = cv2.imread(img_filepath)
        # print('loaded')
        img = img0.copy()  # cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        splits = img_filepath.split("Stage")
        imname = splits[1]
        num = int(os.path.splitext(imname)[0])

        imloc = location(num, dims)
        # print(num,imloc)
        imsmall = cv2.resize(img.copy(), dsize=(int(round(w * rescale, 0)), int(round(h * rescale, 0))))
        offsetw = int(round(w * rescale * imloc[0] * 0.9, 0))
        offseth = int(round(h * rescale * imloc[1] * 0.9, 0))
        # print(offsetw,offseth)
        h2, w2, c2 = imsmall.shape
        maxh = offseth + int(round(h * rescale, 0))
        maxw = offsetw + int(round(w * rescale, 0))
        diffh = int(round(0.1 * h2, 0))
        diffw = int(round(0.1 * w2, 0))
        # print(np.shape(stitch[offseth:maxh,offsetw:maxw]))
        if imloc[1] == 0 and imloc[0] == 0:
            stitch[offseth:maxh, offsetw:maxw] = imsmall
        elif imloc[1] > 0 and imloc[0] == 0:
            stitch[offseth:maxh - diffh, offsetw:maxw] = imsmall[diffh:h2, 0:w2]
            stitch[offseth - diffh:offseth - 1, offsetw:maxw] = (np.int32(imsmall[0:diffh - 1, 0:w2]) + np.int32(
                stitch[offseth - diffh:offseth - 1, offsetw:maxw])) / 2
        elif imloc[0] > 0 and imloc[1] == 0:
            stitch[offseth:maxh, offsetw:maxw - diffw] = imsmall[0:h2, diffw:w2]
            stitch[offseth:maxh, offsetw - diffw:offsetw - 1] = (np.int32(imsmall[0:h2, 0:diffw - 1]) + np.int32(
                stitch[offseth:maxh, offsetw - diffw:offsetw - 1])) / 2
        elif imloc[0] > 0 and imloc[1] > 0:
            stitch[offseth:maxh - diffh, offsetw:maxw - diffw] = imsmall[diffh:h2, diffw:w2]
            stitch[offseth - diffh:offseth - 1, offsetw - diffw:offsetw - 1] = (np.int32(
                imsmall[0:diffh - 1, 0:diffw - 1]) + np.int32(
                stitch[offseth - diffh:offseth - 1, offsetw - diffw:offsetw - 1])) / 2
        tok = time.time()
        print(f"{img_filepath} - {tok - tik} seconds")
    cv2.imwrite(os.path.join(outputloc + "\\pstitch", os.path.basename(str(step) + ".jpg")), stitch)


def location(m, dimset):
    outset = dimset
    height = outset[1]
    width = outset[0]
    row = m % height
    column = (m - row) / height
    # print(m,column,row)
    return column, row


def main(args):
    config = load_queue(args.q or "Queue.txt")
    coordflag = args.map.lower() == "y"

    for input_dir, output_dir in config:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir + "\\pstitch", exist_ok=True)

        files = glob.glob(os.path.join(input_dir, "*"))
        # Filter files to only have images.
        files = [f for f in files if os.path.splitext(f)[1] in [".jpg", ".png", ".jpeg", '.tif']]
        files = [f for f in files if 'Stage' in f]

        testimg = cv2.imread(files[0])
        h, w, c = testimg.shape
        files.sort(key=len)
        # print(files)

        # smuggling output_dir into pool.map by packaging it with the iterable, gets unpacked by run_file_wrapped
        dh = 3648
        dw = int(round(dh * 3 / 2, 0))
        dims = dim_get(input_dir)
        print(dims)
        dimh = int(round(dh * rescale * (0.9 * dims[1] + 0.1), 0)) + 1
        dimw = int(round(dw * rescale * (0.9 * dims[0] + 0.1), 0)) + 1
        # print(dims,dimh,dimw)
        newshape = (dimh, dimw, 3)  # every image except 1 edge overlap
        # print(newshape)
        n_proc = os.cpu_count() - threadsave  # config.jobs if config.jobs > 0 else
        if n_proc > 1:
            fac = max([int(dims[0] / (n_proc - 1)), 1])
            print('fac', fac)
            steps = np.arange(0, dims[0], fac)
            print(steps)
            k = 0
            pararr = []
            farr = []
            while k < len(steps) - 2:
                pararr.append([steps[k], steps[k + 1]])
                k = k + 1
            pararr = (np.array(pararr) * dims[1])
            pararr = np.append(pararr, [[steps[k] * dims[1], dims[0] * dims[1]]], axis=0)
            i = 0
            while i < len(pararr):
                arr = pararr[i]
                # print(arr)
                farr1 = files[arr[0]:arr[1]]
                farr1 = [[f, steps[i], newshape, dims, output_dir] for f in farr1]
                farr.append(farr1)
                i = i + 1
            # print(pararr,len(farr[0]))
            tik = time.time()
            with Pool(n_proc) as pool:
                pool.map(run_file_wrapped, farr)
            tok = time.time()
            pstitches = glob.glob(os.path.join(output_dir + "\pstitch", "*"))
            fstitch = np.int16(np.zeros(newshape))
            plist = []
            for pstitchfile in pstitches:
                splits = pstitchfile.split("pstitch\\")
                imname = splits[1]
                column = int(os.path.splitext(imname)[0])
                pstitch = cv2.imread(pstitchfile)
                plist.append([column, pstitch])
            j = 0
            # print(plist)

            plist.sort(key=lambda x: x[0])
            while j < len(plist):
                p = plist[j]
                pstitch = p[1]
                column = p[0]
                print(column)
                if j < len(plist) - 1:
                    column2 = plist[j + 1][0]
                else:
                    column2 = dims[0]
                h2, w2, c2 = pstitch.shape
                offsetw = int(round(w * rescale * column * 0.9, 0))  # where to offset left edge
                maxw = int(round(w * rescale, 0))  # width of a single small image
                pwidth = int((column2 - column) * maxw)
                delta = 0  # int(round(0.1*maxw,0))
                if column == 0:
                    fstitch[0:h2 - 1, offsetw:offsetw + pwidth] = pstitch[0:h2 - 1, offsetw:offsetw + pwidth]
                else:

                    fstitch[0:h2 - 1, offsetw + delta:offsetw + pwidth] = pstitch[0:h2 - 1,
                                                                          offsetw + delta:offsetw + pwidth]
                    fstitch[0:h2 - 1, offsetw:offsetw + delta - 1] = (np.int32(
                        fstitch[0:h2 - 1, offsetw:offsetw + delta - 1]) + np.int32(
                        fstitch[0:h2 - 1, offsetw:offsetw + delta - 1])) / 2
                j = j + 1
        cv2.imwrite(os.path.join(output_dir, os.path.basename("stitched.jpg")), fstitch)

        if coordflag == 1:
            imlist = np.loadtxt(output_dir + "Imlist.txt", skiprows=1)
            fstitch2 = fstitch.copy()

            sh, sw, sc = fstitch.shape
            for m in imlist:
                coords = location(m, dims)
                coords2 = (int(sw * (coords[0] + 0.1) / (dims[0])), int(sh * (coords[1] + 0.7) / (dims[1])))
                # coords=(int(sw*(coords[0]+0.5)/(dims[0])),int(sh*(coords[1]+0.5)/(dims[1])))
                start = (int(sw * (coords[0]) / (dims[0])), int(sh * (coords[1]) / (dims[1])))
                end = (int(sw * (coords[0] + 1) / (dims[0])), int(sh * (coords[1] + 1) / (dims[1])))
                fstitch2 = img4 = cv2.putText(fstitch2, str(int(m)), coords2, cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                              (0, 0, 255), 2, cv2.LINE_AA)
                fstitch2 = cv2.rectangle(fstitch2, start, end, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(output_dir, os.path.basename("coordmap.jpg")), fstitch2)

        print(f"Total for {len(files)} files: {tok - tik} = avg of {(tok - tik) / len(files)} per file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find graphene flakes on SiO2. Currently configured only for exfoliator dataset"
    )
    parser.add_argument(
        "--q",
        required=False,
        type=str,
        help="Queue file with list of IO directories"
    )
    parser.add_argument(
        "--map",
        required=True,
        type=str,
        help="Does Imlist.txt exist, allowing flake locations to be described? (y/N)"
    )
    args = parser.parse_args()
    main(args)
