import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import re
import argparse
import os
from util.logger import logger
from util.queue import load_queue


Dimensions = tuple[int, int]


def pos_get(input_dir: str) -> np.ndarray[float]:
    """
    Gets the scan dimensions and positions contained in a leica metadata file.
    :param input_dir: The directory containing the microscope file.
    :return: A 2d numpy array mapping [y dimension, x dimension] to [y position (mm), x position (mm)]
    """
    with open(input_dir + "/leicametadata/TileScan_001.xlif", 'r') as file:
        rawdata = file.read()

    pos_arr = np.zeros((100, 100, 2))

    # Match metadata <Tile> tags in the form of <Tile FieldX="..." FieldY="..." PosX="..." PosY="..." PosZ="..." />
    matches = re.findall(r"<Tile FieldX=\"(.+)\" FieldY=\"(.+)\" PosX=\"(.+)\" PosY=\"(.+)\" PosZ=\".+\" />", rawdata)
    for xd, yd, posx, posy in matches:
        pos_arr[int(yd), int(xd)] = np.array([float(posy), float(posx)])

    return pos_arr
def mag_get(input_dir: str) -> str:
    with open(input_dir + "/leicametadata/TileScan_001.xlif", 'r') as file:
        rawdata = file.read()
    rawdata=rawdata.split('ObjectiveName=')[1]
    rawdata=rawdata.split('ObjectiveNumber')[0]
    #print(rawdata)
    if '5x' in rawdata:
        return '5x'
    elif '10x' in rawdata:
        return '10x'

def dim_get(input_dir: str) -> Dimensions:
    """
    Gets the scan dimensions from a microscope file.
    :param input_dir: The directory containing the microscope file.
    :return: The dimensions of the scan, as a tuple of [width, height].
    """
    data = pos_get(input_dir)

    points = np.nonzero(data)
    xmax = np.max(points[1])
    ymax = np.max(points[0])

    return xmax + 1, ymax + 1


def location(m: int, dims: Dimensions) -> tuple[float, int]:  # TODO: supposed to be a float?
    """
    Gets the (x, y) location of the mth scan image given the width and height of the scan.
    :param m: The index to locate.
    :param dims: The dimensions of the scan, as a tuple of (width, height).
    :return: The location, as a tuple of (x, y).
    """
    width, height = dims
    row = m % height
    column = (m - row) / height

    return column, row


def get_stage(path: str) -> int:
    """
    Gets the stage number of a scan image given the file path.
    :param path: The file path (ex. `C:/.../TileScan_001--Stage250.jpg`)
    :return: The stage number (ex. `250`)
    
    """
    try:
        stage=int(re.search(r"Stage(\d{3,4})", path).group(1))
    except:
        try:
            stage=int(re.search(r"stage(\d{3,4})", path).group(1))
        except:
            stage=int(os.path.splitext(path)[0].split('\\')[-1])
    return stage


if __name__ == '__main__':
    # TODO: new description or abstract
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

    for input_dir, _ in load_queue(args.q):
        logger.info(f"Found dimensions for {input_dir}: {dim_get(input_dir)}")
