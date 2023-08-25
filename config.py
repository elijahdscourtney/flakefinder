import cv2
import numpy as np


threadsave = 4  # number of threads NOT allocated when running
boundflag = 1

k = 4

# Detection parameters
FLAKE_MIN_AREA_UM2 = 200
MULTILAYER_FLAKE_MIN_AREA_UM2 = 50
FLAKE_MAX_AREA_UM2 = 6000

COLOR_WINDOW=(5,5,5)
COLOR_CHECK_OFFSETUM = 10
COLOR_RATIO_CUTOFF=0.1 #fraction of imchunk +- offset that must look like a color to be counted for sorting purposes
COLOR_PASS_CUTOFF=0.3 #fraction of imchunk that must look like flake to be passed on

FLAKE_MIN_EDGE_LENGTH_UM = 10
FLAKE_ANGLE_TOLERANCE_RADS = np.deg2rad(4)
FLAKE_R_CUTOFF = 80
epsratio=0.05

OPEN_MORPH_SHAPE = cv2.MORPH_RECT
CLOSE_MORPH_SHAPE = cv2.MORPH_CROSS
EQUALIZE_OPEN_MORPH_SHAPE = cv2.MORPH_CROSS
EQUALIZE_CLOSE_MORPH_SHAPE = cv2.MORPH_RECT

BOX_RGB = (255, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
#10x, 5x
UM_TO_PXs = [4.164,2.082]
t_color_match_counts = [0.000225*FLAKE_MIN_AREA_UM2/200, 0.25*0.000225*FLAKE_MIN_AREA_UM2/200]  # fraction of image that must look like monolayers
OPEN_MORPH_SIZES = [2,1]
CLOSE_MORPH_SIZES = [6,1]
EQUALIZE_OPEN_MORPH_SIZES = [3,1]
EQUALIZE_CLOSE_MORPH_SIZES = [2,1]
maxlinegaps=[6,3]
BOX_OFFSETS = [5,2]
BOX_THICKNESSES = [6,3]



