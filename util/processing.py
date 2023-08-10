import cv2
import numpy as np

from config import OPEN_MORPH_SIZE, CLOSE_MORPH_SIZE, OPEN_MORPH_SHAPE, CLOSE_MORPH_SHAPE, UM_TO_PX, \
                   FLAKE_MIN_EDGE_LENGTH_UM, FLAKE_ANGLE_TOLERANCE_RADS, k

RGB = tuple[int, int, int]
FlakeRGB = np.ndarray[int]


def bg_to_flake_color(rgb: RGB) -> FlakeRGB:
    """
    Returns the flake color based on an input background color. Values determined empirically.
    :param rgb: The RGB array representing the color of the background.
    :return: The RGB array representing the color of the flake.
    """
    red, green, blue = rgb

    flake_red = 0.8289 * red +1.888
    flake_green = 0.9456 * green + 0.402
    flake_blue = 1.061 * blue-4.336

    return np.array([flake_red, flake_green, flake_blue])


def get_bg_pixels(img: np.ndarray):
    # Lower and higher RGB limits for what code can see as background
    # lowlim = np.array([87, 100, 99])
    # highlim = np.array([114, 118, 114])

    imsmall = cv2.resize(img.copy(), dsize=(256 * k, 171 * k)).reshape(-1, 3)
    # test = np.sign(imsmall - lowlim) + np.sign(highlim - imsmall)
    return imsmall # * np.sign(test + abs(test))


def get_avg_rgb(img: np.ndarray, mask: np.ndarray[bool] = 1) -> RGB:
    """
    Gets the average RGB within a given array of RGB values.
    :param img: The image to process.
    :param mask: An optional mask to apply to RGB values.
    :return: The average RGB.
    """
    red_freq = np.bincount(img[:, 0] * mask)
    green_freq = np.bincount(img[:, 1] * mask)
    blue_freq = np.bincount(img[:, 2] * mask)

    red_freq[0] = 0  # otherwise argmax finds values masked to 0
    green_freq[0] = 0
    blue_freq[0] = 0
    
    return int(red_freq.argmax()), int(green_freq.argmax()), int(blue_freq.argmax())

def mask_bg(img: np.ndarray, back_rgb: tuple[int, int, int], back_hsv: tuple[int, int, int]) -> np.ndarray:
    lower = tuple(map(int,  np.array(back_rgb) -(44, 15, 5)))
    higher = tuple(map(int, np.array(back_rgb) -(8, -24, -8)))
    #print(lower,higher)
    maskrgb=cv2.inRange(img, lower, higher)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    lower = tuple(map(int,  np.array(back_hsv) - (75,30,19)))
    higher = tuple(map(int, np.array(back_hsv) - (-120,-60,-7)))
    maskhsv=cv2.inRange(img_hsv, lower, higher)
    maskbg=maskrgb*maskhsv.astype(np.float32)/255
    return maskbg.astype(np.uint8)
def mask_flake_color(img: np.ndarray, flake_avg_hsv: np.ndarray) -> np.ndarray:
    """
    Mask an image to black and white pixels based on whether it is within threshold of the given flake color, used by
    the algorithm in `monolayer_finder.py`.

    :param flake_avg_hsv: The average flake color (in HSV).
    :param img: The RGB image.
    :return: The black and white image mask.
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower = tuple(map(int, flake_avg_hsv - (6, 25, 25)))
    higher = tuple(map(int, flake_avg_hsv + (6, 25, 25)))
    
    mask=cv2.inRange(img_hsv, lower, higher)
    return mask

def mask_flake_color2(img: np.ndarray, flake_avg_rgb: np.ndarray) -> np.ndarray:
    """
    Mask an image to black and white pixels based on whether it is within threshold of the given flake color, used by
    the algorithm in `monolayer_finder.py`.

    :param flake_avg_hsv: The average flake color (in HSV).
    :param img: The RGB image.
    :return: The black and white image mask.
    """
    mask = np.sqrt(np.sum((img - flake_avg_rgb) ** 2, axis=2))
    mask=np.asarray(mask<8)
    lower = tuple(map(int, flake_avg_rgb - (8, 7, 5)))
    higher = tuple(map(int, flake_avg_rgb + (8, 7, 5)))
    mask2=cv2.inRange(img, lower, higher)
    mask=mask*mask2
    return mask

def is_edge_image(img):
    """
    Gets whether an image lies on the edge of the scan.
    :param img: The image to check.
    :return: Whether it lies on the edge of the scan (whether there are too many dark pixels).
    """
    img_h, img_w, _ = img.shape
    img_pixels = img_h * img_w

    mask = cv2.inRange(img, (0, 0, 0), (50, 50, 50))
    return cv2.countNonZero(mask) / img_pixels > 0.1


def mask_equalized(equalized: np.ndarray) -> np.ndarray:
    """
    Mask for dark pixels in an equalized grayscale image. Used to filter flakes for the algorithm in
    `equalize_monolayer_finder.py`.

    :param equalized: The grayscale result of `cv2.equalizeHist()`.
    :return: The black and white image mask.
    """
    _, equalize_mask = cv2.threshold(equalized, 25, 255, cv2.THRESH_BINARY_INV)
    return equalize_mask


def mask_outer(img_hsv: np.ndarray, back_hsv: tuple[int, int, int]) -> np.ndarray:
    is_special = 35 < back_hsv[0] < 50 or back_hsv[1] > 50  # TODO: rather hacky
    return cv2.inRange(
        img_hsv,
        (82 if is_special else 90, int(back_hsv[1]) + (-10 if is_special else 20), 105),
        (105, int(back_hsv[1]) + 50, int(back_hsv[2]) + 5)
    )


def apply_morph_open(masked: np.ndarray, size: int = OPEN_MORPH_SIZE, shape=OPEN_MORPH_SHAPE) -> np.ndarray:
    """
    Applies the "opening" morphological operation to a masked image to clear away small false-positive "islands".
    https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html

    :param masked: The masked black and white image from `mask_flake_color`.
    :param size: The size of the transform.
    :param shape: The structuring element shape of the transform.
    :return: The black and white image, with the morph applied.
    """
    element = cv2.getStructuringElement(shape, (2 * size + 1, 2 * size + 1))
    return cv2.morphologyEx(masked, cv2.MORPH_OPEN, element)


def apply_morph_close(masked: np.ndarray, size: int = CLOSE_MORPH_SIZE, shape=CLOSE_MORPH_SHAPE) -> np.ndarray:
    """
    Applies the "closing" morphological operation to a masked image to fill small "holes" in detected flakes.
    https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html

    :param masked: The masked black and white image from `mask_flake_color`.
    :param size: The size of the transform.
    :param shape: The structuring element shape of the transform.
    :return: The black and white image, with the morph applied.
    """
    element = cv2.getStructuringElement(shape, (2 * size + 1, 2 * size + 1))
    return cv2.morphologyEx(masked, cv2.MORPH_CLOSE, element)


def in_bounds(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> bool:
    """
    Gets if a flake bounded by (x1, y1) and (x2, y2) is entirely contained in another image.

    :param x1: The lower-left x coordinate.
    :param y1: The lower-left y coordinate.
    :param x2: The upper-right x coordinate.
    :param y2: The upper-right y coordinate.
    :param w: The width of the box.
    :param h: The height of the box.
    :return: Whether the flake is entirely contained in another image.
    """
    delt = 0.05
    return x2 > delt * w and y2 > delt * h and x1 < (1 - delt) * w and y1 < (1 - delt) * h


def get_lines(img: np.ndarray, contour):# -> np.ndarray[tuple[tuple[float, float, float, float]]] | None:
    mask = np.zeros(img.shape, np.uint8)
    mask = cv2.drawContours(mask, contour, -1, (255, 255, 255), 1)

    # TODO: make the mask b&w to begin with
    # https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
    return cv2.HoughLinesP(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), 1, np.pi / 180, 50, None, FLAKE_MIN_EDGE_LENGTH_UM * UM_TO_PX, 5)


def get_angles(lines: np.ndarray[tuple[tuple[float, float, float, float]]]) -> list[float]:
    """
    Gets all angles within a given range of a multiple of 30 degrees (excluding 180 and 360) given a list of lines.
    :param lines: The lines to get angles from (from `HoughLinesP`, as tuples of [x1, y1, x2, y2]).
    :return: The list of filtered angles (in radians).
    """
    ret = []

    for i in range(0, len(lines)):
        for j in range(i, len(lines)):
            x11, y11, x21, y21 = lines[i][0]
            x12, y12, x22, y22 = lines[j][0]

            # Calculate angle between lines
            t1 = np.arctan2(x21 - x11, y21 - y11)
            t2 = np.arctan2(x22 - x12, y22 - y12)
            t = (t2 - t1) % (2 * np.pi)

            if t % (np.pi / 6) > FLAKE_ANGLE_TOLERANCE_RADS or t < (np.pi / 6) - FLAKE_ANGLE_TOLERANCE_RADS or t > 2 * np.pi - FLAKE_ANGLE_TOLERANCE_RADS:
                continue

            ret.append(t)

    return ret
