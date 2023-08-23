import cv2
import numpy as np


threadsave = 4  # number of threads NOT allocated when running
boundflag = 1

k = 4

# Detection parameters
#UM_TO_PX = 4.164 #10x
UM_TO_PX = 2.082 #5x


FLAKE_MIN_AREA_UM2 = 200
MULTILAYER_FLAKE_MIN_AREA_UM2 = 50
t_color_match_count = 0.000225*FLAKE_MIN_AREA_UM2/200  # fraction of image that must look like monolayers
FLAKE_MAX_AREA_UM2 = 6000


COLOR_WINDOW=(6,6,6)
COLOR_CHECK_OFFSETUM = 10
COLOR_RATIO_CUTOFF=0.05 #fraction of imchunk +- offset that must look like a color to be counted for sorting purposes
COLOR_PASS_CUTOFF=0.3 #fraction of imchunk that must look like flake to be passed on

FLAKE_MIN_EDGE_LENGTH_UM = 10
FLAKE_ANGLE_TOLERANCE_RADS = np.deg2rad(4)

FLAKE_R_CUTOFF = 80

# Morphology parameters
OPEN_MORPH_SIZE = 2
CLOSE_MORPH_SIZE = 6

epsratio=0.05

OPEN_MORPH_SHAPE = cv2.MORPH_RECT
CLOSE_MORPH_SHAPE = cv2.MORPH_CROSS

EQUALIZE_OPEN_MORPH_SIZE = 3
EQUALIZE_CLOSE_MORPH_SIZE = 2

EQUALIZE_OPEN_MORPH_SHAPE = cv2.MORPH_CROSS
EQUALIZE_CLOSE_MORPH_SHAPE = cv2.MORPH_RECT

# Labelling parameters
BOX_OFFSET = 5
BOX_RGB = (255, 0, 0)
BOX_THICKNESS = 6

FONT = cv2.FONT_HERSHEY_SIMPLEX
