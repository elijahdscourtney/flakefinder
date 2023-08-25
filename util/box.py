from dataclasses import dataclass

import cv2
import numpy as np

from config import UM_TO_PXs, FLAKE_MIN_AREA_UM2, FLAKE_MAX_AREA_UM2, FLAKE_R_CUTOFF, BOX_OFFSETS, BOX_THICKNESSES, FONT, BOX_RGB, epsratio, COLOR_WINDOW, COLOR_RATIO_CUTOFF, COLOR_CHECK_OFFSETUM
from util.processing import in_bounds, get_angles, get_avg_rgb, bg_to_flake_color, apply_morph_close
from config import CLOSE_MORPH_SIZES, CLOSE_MORPH_SHAPE, MULTILAYER_FLAKE_MIN_AREA_UM2

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
def approxpolygon(cnt,epsratio):
    epsilon=epsratio*(cv2.arcLength(cnt,True))
    approx=cv2.approxPolyDP(cnt,epsilon,True)
    return approx

def make_boxes(contours, hierarchy, img_h: int, img_w: int, magx: str) -> list[list[Box],list[float]]:
    """
    Make boxes from contours, filtering out contours that are too small or completely contained by another image.
    :param contours: The contours to draw boxes from.
    :param hierarchy: The hierarchy of contours, as returned from `findContours()` with return mode `RETR_TREE`.
    :param img_h: The height of the image.
    :param img_w: The width of the image.
    :return: The list of boxes.
    """
    if magx=='5x':
        UM_TO_PX=UM_TO_PXs[1]
    elif magx=='10x':
        UM_TO_PX=UM_TO_PXs[0]
    boxes = []
    inner_indices = []
    flake_rs=[]
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
        #approx=approxpolygon(cnt,epsratio)
        #img=np.zeros((img_h,img_w,3))
        #for i in range(len(cnt)):
            #img=cv2.drawContours(img, [cnt[i]], -1, (0,255,0), 3)
        #perimeter=cv2.arcLength(approx, True)
        if area < FLAKE_MIN_AREA_UM2 * (UM_TO_PX ** 2):
            continue
        if area > FLAKE_MAX_AREA_UM2 * (UM_TO_PX ** 2):
            continue 
        flake_r=(perimeter ** 2) / area
        if  flake_r> FLAKE_R_CUTOFF:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if not in_bounds(x, y, x + w, y + h, img_w, img_h):
            continue
        flake_rs.append(flake_r)
        boxes.append(Box(cnt, area, x, y, w, h))
    return [boxes,flake_rs]


def merge_boxes(boxes: list[Box], flake_rs: list) -> list[list[Box],list[float]]:
    """
    Merges a list of boxes by combining boxes with overlap.
    :param boxes: The list of boxes to merge.
    :return: The merged list.
    """
    merged = []
    eliminated_indexes = []
    merged_rs=[]
    for _i in range(len(boxes)):
        if _i in eliminated_indexes:
            continue
        i = boxes[_i]
        avg_r=flake_rs[_i]
        for _j in range(_i + 1, len(boxes)):
            j = boxes[_j]
            rj=flake_rs[_j]
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
                avg_r=(avg_r+rj)/2
                eliminated_indexes.append(_j)
        merged.append(i)
        merged_rs.append(avg_r)
    return [merged,merged_rs]


def draw_box(img: np.ndarray, b: Box, magx: str) -> np.ndarray:
    """
    Labels a box on an image, drawing the bounding rectangle and labelling the micron height and width.
    :param img: The image to label.
    :param b: The box to label.
    :return: The labelled image.
    """
    # microns/pixel from Leica calibration
    if magx=='5x':
        BOX_OFFSET=BOX_OFFSETS[1]
        UM_TO_PX=UM_TO_PXs[1]
        BOX_THICKNESS=BOX_THICKNESSES[1]
    elif magx=='10x':
        BOX_OFFSET=BOX_OFFSETS[0]
        UM_TO_PX=UM_TO_PXs[0]
        BOX_THICKNESS=BOX_THICKNESSES[0]
    x = int(b.x) - BOX_OFFSET
    y = int(b.y) - BOX_OFFSET
    w = int(b.width) + 2 * BOX_OFFSET
    h = int(b.height) + 2 * BOX_OFFSET

    width_microns = int(round(w/UM_TO_PX, 0))
    height_microns = int(round(h/UM_TO_PX, 0))  # microns

    img = cv2.rectangle(img, (x, y), (x + w, y + h), BOX_RGB, BOX_THICKNESS)
    img = cv2.putText(img, str(height_microns)+'um', (x + w + 10, y + int(h / 2)), FONT, 1, (0, 0, 0), 2, cv2.LINE_AA)
    img = cv2.putText(img, str(width_microns)+'um', (x, y - 10), FONT, 1, (0, 0, 0), 2, cv2.LINE_AA)

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

def draw_line_angles(img: np.ndarray, box: Box, lines) -> list[[float,str]]:
    labeledangles = []
    labels=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O']
    linelabels=[]
    if lines is not None:
        i=0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (192, 8, 254), 2, cv2.LINE_AA)
            
            label=labels[i%len(labels)]
            i=i+1
            offset=20
            thet=np.arctan((y2-y1)/(x2-x1))
            yoff=int(offset*np.cos(thet))
            xoff=int(offset*np.sin(thet))
            linelabels.append([line,label])
            cv2.putText(img, label,
                        (int((x2+x1)/2) - xoff, int((y2+y1)/2) + yoff),
                        FONT, 2/3, (0, 0, 0), 2, cv2.LINE_AA) 
        if len(linelabels) < 2:
            return labeledangles

        labeledangles = get_angles(linelabels)
    return labeledangles
def label_angles(img: np.ndarray, labeledangles: list[[float,str]], box: Box) -> np.ndarray:
    for i in range(len(labeledangles)):
            angle=labeledangles[i][0]
            label=labeledangles[i][1]
            goodangle=np.min([angle,abs(angle-2*np.pi)])
            cv2.putText(img, label+':'+str(int(round(np.rad2deg(goodangle), 0))) + ' deg.',
                        (box.x + box.width + 10, box.y + int(box.height / 2) + (i + 1) * 35),
                        FONT, 1, (0, 0, 0), 2, cv2.LINE_AA) 
    return img
def get_color_ratio(img0:np.ndarray, b: Box, color: tuple[int,int,int], mode:str, magx: str) ->float:
    #print(color)
    color=np.array(color)
    h,w,c=img0.shape
    if magx=='5x':
        UM_TO_PX=UM_TO_PXs[1]
    elif magx=='10x':
        UM_TO_PX=UM_TO_PXs[0]
    if mode=='typecheck':
        d=int(COLOR_CHECK_OFFSETUM*UM_TO_PX)
    if mode=='sizecheck':
        d=0
    x1 = np.max([0,b.x-d])
    x2 = np.min([b.x + b.width+d,w])
    y1 = np.max([0,b.y-d])
    y2 = np.min([b.y + b.height+d,h])
    barea=(y2-y1)*(x2-x1)
    img=img0[y1:y2,x1:x2]
    
    maskrgb=cv2.inRange(img, color-np.array(COLOR_WINDOW), color+np.array(COLOR_WINDOW))
    maskrgb=apply_morph_close(maskrgb,magx,sizes=CLOSE_MORPH_SIZES, shape=CLOSE_MORPH_SHAPE)
    #cv2.imshow('imc',maskrgb)
    #cv2.waitKey(0)
    masksum=np.sum(maskrgb)/255
    
    return masksum/barea, barea

def check_color_ratios(img:np.ndarray, b: Box, back_rgb: tuple[int, int, int], real_flake_rgb: tuple[int, int, int], magx: str) -> str:
    if magx=='5x':
        UM_TO_PX=UM_TO_PXs[1]
    elif magx=='10x':
        UM_TO_PX=UM_TO_PXs[0]
    R1=COLOR_RATIO_CUTOFF
    back_rgb=np.array(back_rgb)
    real_flake_rgb=np.array(real_flake_rgb)
    monolayer_rgb=real_flake_rgb
    bilayer_rgb=bg_to_flake_color(back_rgb,2)
    trilayer_rgb=bg_to_flake_color(back_rgb,3)
    bg_ratio,barea=get_color_ratio(img,b,back_rgb,'typecheck', magx)
    mono_ratio,barea=get_color_ratio(img,b,monolayer_rgb,'typecheck', magx)
    bi_ratio,barea=get_color_ratio(img,b,bilayer_rgb,'typecheck', magx)
    tri_ratio,barea=get_color_ratio(img,b,trilayer_rgb,'typecheck', magx)
    remainder=1-bg_ratio-mono_ratio-bi_ratio-tri_ratio
    print('BG','Mono','Bi','Tri','Other')
    print(bg_ratio,mono_ratio,bi_ratio,tri_ratio,remainder)
    img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h,w,c=img.shape
    Rmono=(FLAKE_MIN_AREA_UM2*UM_TO_PX**2)/barea #making sure it's large enough
    Rmult=(MULTILAYER_FLAKE_MIN_AREA_UM2*UM_TO_PX**2)/barea
    if remainder>R1:
        return 'Bulk'
    elif tri_ratio>Rmult:
        return 'Trilayer'
    elif bi_ratio>Rmult:
        return 'Bilayer'
    elif mono_ratio>R1 or mono_ratio>Rmono:
        return 'Monolayer'
    else:
        return 'Bulk'