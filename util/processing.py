import cv2
import numpy as np
from sklearn.cluster import DBSCAN

RGB = list[int]
FlakeRGB = np.ndarray[int]


def bg_to_flake_color(rgb: RGB) -> FlakeRGB:
    """
    Returns the flake color based on an input background color. Values determined empirically.
    :param rgb: The RGB array representing the color of the background.
    :return: The RGB array representing the color of the flake.
    """
    red, green, blue = rgb

    flake_red = int(round(0.8643 * red - 2.55, 0))
    flake_green = int(round(0.8601 * green + 9.6765, 0))
    flake_blue = blue + 4

    return np.array([flake_red, flake_green, flake_blue])


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

    return [red_freq.argmax(), green_freq.argmax(), blue_freq.argmax()]


def find_chunks(dbscan_img, t_min_cluster_pixel_count, t_max_cluster_pixel_count) -> tuple[np.ndarray, np.ndarray]:
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
    criteria = (n_pixels > t_min_cluster_pixel_count) & (n_pixels < t_max_cluster_pixel_count)

    h_labels = np.array(label_names)[criteria]
    h_labels = h_labels[h_labels > 0]

    return labels, h_labels


# this identifies the edges of flakes, resource-intensive but useful for determining if flake ID is working
def edgefind(imchunk: np.ndarray, avg_rgb: FlakeRGB, pixcals: list[float], t_rgb_dist: int) -> tuple[RGB, any, float]:  # TODO
    """
    TODO
    :param imchunk: The image chunk to find edges in.
    :param avg_rgb: The average flake RGB, from `bg_to_flake_color()`.
    :param pixcals:
    :param t_rgb_dist: The threshold a pixel color must be within from the average flake color to be counted as good.
    :return: The results, as a tuple of (flake rgb, edge image, flake area).
    """
    pixcalw, pixcalh = pixcals
    dims = np.shape(imchunk)

    impix = imchunk.copy().reshape(-1, 3)
    flakeid = np.sqrt(np.sum((impix - avg_rgb) ** 2, axis=1)) < t_rgb_dist  # a mask for pixel color

    # determines flake RGB as the most common R,G,B value in identified flake region
    rgb = get_avg_rgb(impix, flakeid)

    h, w, c = imchunk.shape
    flakeid2 = np.sqrt(np.sum((impix - rgb) ** 2, axis=1)) < 5  # a mask for pixel color
    maskpic2 = np.reshape(flakeid2, (dims[0], dims[1], 1))
    indices = np.argwhere(np.any(maskpic2 > 0, axis=2))  # flake region
    farea = round(len(indices) * pixcalw * pixcalh, 1)

    grayimg = cv2.cvtColor(imchunk, cv2.COLOR_BGR2GRAY)
    grayimg = cv2.fastNlMeansDenoising(grayimg, None, 2, 3, 11)
    edgeim = np.reshape(cv2.Canny(grayimg, 5, 15), (h, w, 1)) \
               .astype(np.int16) * np.array([25, 25, 25]) / 255

    return rgb, edgeim.astype(np.uint8), farea
