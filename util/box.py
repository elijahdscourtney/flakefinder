from dataclasses import dataclass

import cv2
import numpy as np

from config import UM_TO_PX, FLAKE_MIN_AREA_UM2, FLAKE_MAX_AREA_UM2, FLAKE_R_CUTOFF, BOX_OFFSET, BOX_THICKNESS, FONT, BOX_RGB
from util.processing import in_bounds, get_angles, get_avg_rgb


@dataclass
class Box:
    contours: np.ndarray
    area: float
    x: int
    y: int
    width: int
    height: int

    def intersects(self, other: 'Box', b=5) -> bool:
        x1 = self.x
        x2 = self.x + self.width
        y1 = self.y
        y2 = self.y + self.height

        x3 = other.x - b
        x4 = other.x + other.width + b
        y3 = other.y - b
        y4 = other.y + other.height + b

        return y1 <= y4 and y2 >= y3 and x1 <= x4 and x2 >= x3


def make_boxes(contours, hierarchy, img_h: int, img_w: int) -> list[Box]:
    """
    Make boxes from contours, filtering out contours that are too small or completely contained by another image.
    :param contours: The contours to draw boxes from.
    :param hierarchy: The hierarchy of contours, as returned from `findContours()` with return mode `RETR_TREE`.
    :param img_h: The height of the image.
    :param img_w: The width of the image.
    :return: The list of boxes.
    """
    boxes = []
    inner_indices = []
    for i in range(len(contours)):
        if i in inner_indices:
            continue

        cnt = contours[i]
        _, _, child, parent = hierarchy[0][i]

        area = cv2.contourArea(cnt)

        perimeter = cv2.arcLength(cnt, True)

        # Subtract child contours to better represent area
        # https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
        # TODO: recursion to children of children?
        while child != -1:
            child_cnt = contours[child]
            area -= cv2.contourArea(child_cnt)

            cnt = np.concatenate([cnt, child_cnt])  # Add the child contour to `cnt` so it shows up in the final image
            inner_indices.append(child)

            # Move to next contour on same level as defined by hierarchy tree, if it exists
            child, _, _, _ = hierarchy[0][child]

        if area < FLAKE_MIN_AREA_UM2 * (UM_TO_PX ** 2):
            continue
        if area > FLAKE_MAX_AREA_UM2 * (UM_TO_PX ** 2):
            continue 

        if (perimeter ** 2) / area > FLAKE_R_CUTOFF:
            
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if not in_bounds(x, y, x + w, y + h, img_w, img_h):
            continue

        boxes.append(Box(cnt, area, x, y, w, h))

    return boxes


def merge_boxes(boxes: list[Box]) -> list[Box]:
    """
    Merges a list of boxes by combining boxes with overlap.
    :param boxes: The list of boxes to merge.
    :return: The merged list.
    """
    merged = []
    eliminated_indexes = []

    for _i in range(len(boxes)):
        if _i in eliminated_indexes:
            continue
        i = boxes[_i]
        for _j in range(_i + 1, len(boxes)):
            j = boxes[_j]

            if i.intersects(j):
                # Extend the first box to include dimensions of the 2nd box.
                x_min = min(i.x, j.x)
                x_max = max(i.x + i.width, j.x + j.width)
                y_min = min(i.y, j.y)
                y_max = max(i.y + i.height, j.y + j.height)

                new_width = x_max - x_min
                new_height = y_max - y_min
                i = Box(
                    np.concatenate([i.contours, j.contours]),
                    i.area + j.area,
                    x_min, y_min,
                    new_width, new_height
                )
                eliminated_indexes.append(_j)
        merged.append(i)

    return merged


def draw_box(img: np.ndarray, b: Box) -> np.ndarray:
    """
    Labels a box on an image, drawing the bounding rectangle and labelling the micron height and width.
    :param img: The image to label.
    :param b: The box to label.
    :return: The labelled image.
    """
    pixcal = 1314.08 / img.shape[1]  # microns/pixel from Leica calibration

    x = int(b.x) - BOX_OFFSET
    y = int(b.y) - BOX_OFFSET
    w = int(b.width) + 2 * BOX_OFFSET
    h = int(b.height) + 2 * BOX_OFFSET

    width_microns = round(w * pixcal, 1)
    height_microns = round(h * pixcal, 1)  # microns

    img = cv2.rectangle(img, (x, y), (x + w, y + h), BOX_RGB, BOX_THICKNESS)
    img = cv2.putText(img, str(height_microns), (x + w + 10, y + int(h / 2)), FONT, 1, (0, 0, 0), 2, cv2.LINE_AA)
    img = cv2.putText(img, str(width_microns), (x, y - 10), FONT, 1, (0, 0, 0), 2, cv2.LINE_AA)

    return img

def get_flake_color(img: np.ndarray, flakergb: np.ndarray, b: Box) -> np.ndarray:
    x = int(b.x)
    y = int(b.y)
    w = int(b.width)
    h = int(b.height)
    imchunk=img[y:y+h,x:x+w]
    lower = tuple(map(int, flakergb - (6, 6, 6)))
    higher = tuple(map(int, flakergb + (6, 6, 6)))
    masker=cv2.inRange(imchunk, lower, higher)/255
    h,w,c=imchunk.shape
    imchunk=imchunk.reshape((-1,3))
    
    masker=masker.reshape((-1,1)).astype(np.uint8)
    imchunk2=imchunk*masker
    rgb=get_avg_rgb(imchunk2)
    if np.array(rgb).all()>0:
        return rgb
    else:
        rgb=[np.bincount(imchunk[:, 0]).argmax(),np.bincount(imchunk[:, 1]).argmax(),np.bincount(imchunk[:, 2]).argmax()]
        return rgb

def draw_line_angles(img: np.ndarray, box: Box, lines) -> list[float]:
    angles = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (192, 8, 254), 2, cv2.LINE_AA)

        if len(lines) < 2:
            return angles

        angles = get_angles(lines)
        for i in range(len(angles)):
            cv2.putText(img, str(round(np.rad2deg(angles[i]), 2)) + ' deg.',
                        (box.x + box.width + 10, box.y + int(box.height / 2) + (i + 1) * 35),
                        FONT, 1, (0, 0, 0), 2, cv2.LINE_AA)

    return angles
