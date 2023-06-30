"""
Note: Currently only configured for Exfoliator tilescans. Very unlikely to work well on other datasets.
"""
import glob
import os
from sklearn.cluster import DBSCAN
from multiprocessing import Pool
import argparse
import matplotlib
matplotlib.use('tkagg')
from matplotlib.patches import Rectangle
from dataclasses import dataclass
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
from functools import reduce
from scipy.spatial import ConvexHull


def imread(path):
    raw = cv2.imread(path)
    return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

@dataclass
class Box:
    label: str
    x: int
    y: int
    width: int
    height: int

    def to_mask(self, img, b=5):
        h, w = img.shape
        boundx=min(0,self.x)
        boundy=min(0,self.y)
        return np.logical_and.outer(
            np.logical_and(np.arange(boundy,h) >= self.y - b, np.arange(boundy,h) <= self.y + self.height + 2 * b),
            np.logical_and(np.arange(boundx,w) >= self.x - b, np.arange(boundx,w) <= self.x + self.width + 2 * b),
        )

flake_colors_rgb = [
    # # Thick-looking
    # [6, 55, 94],
    # Monolayer-looking
     #[57, 65, 86],
     #[60, 66, 85],
    #[89,99,109],
    [0,0,0],
]
flake_colors_hsv = [
    np.uint8(matplotlib.colors.rgb_to_hsv(x) * np.array([179, 255, 255])) for x in flake_colors_rgb
    #matplotlib outputs in range 0,1. Opencv expects HSV images in range [0,0,0] to [179, 255,255]
]
flake_color_hsv = np.mean(flake_colors_hsv, axis=0)
avg_rgb = np.mean(flake_colors_rgb, axis=0)

output_dir = 'cv_output'
threadsave=8 #number of threads NOT allocated when running
boundflag=1
t_rgb_dist = 8
#t_hue_dist = 12 #12
t_red_dist = 12
#t_red_cutoff = 0.1 #fraction of the chunked image that must be more blue than red to be binned
t_color_match_count = 0.000225 #fraction of image that must look like monolayers
k = 4
t_min_cluster_pixel_count = 30*(k/4)**2  # flake too small
t_max_cluster_pixel_count = 20000*(k/4)**2  # flake too large
 # scale factor for DB scan. recommended values are 3 or 4. Trade-off in time vs accuracy. Impact epsilon.
scale=1 #the resolution images are saved at, relative to the original file. Does not affect DB scan
uderivlim=np.array([5,7,30]) #bgr
lderivlim=np.array([0,0,0])
graylim=5
offset=3
# This would be a decorator but apparently multiprocessing lib doesn't know how to serialize it.
def derivget(img,direction,offset): #takes imgand subtracts itself by offset in direction
    absoffset=abs(offset)
    noiseh=2
    
    if len(np.shape(img))==2:
        c=1
        h,w=np.shape(img)
        denoisedimg=cv2.fastNlMeansDenoising(img,None,noiseh,3,11) 
    else:
        h,w,c=img.shape
        denoisedimg=cv2.fastNlMeansDenoisingColored(img,None,noiseh,10,3,11) 
    
    #cv2.imshow('noise',denoisedimg)
    #cv2.waitKey(0)
    img=denoisedimg.astype(np.int32)
    if direction=='x': #left to right deriv
        d1=img[0:h,0:w-absoffset]
        d2=img[0:h,absoffset:w]
    elif direction=='y':
        d1=img[0:h-absoffset,0:w]
        d2=img[absoffset:h,0:w]
    delt=abs(d2-d1)/absoffset
    #print(delt)
    delt=delt.astype(np.uint8)
    return delt

def derivget2(img,direction,offset):
    absoffset=abs(offset)
    noiseh=2
    
    if len(np.shape(img))==2:
        c=1
        
        h,w=np.shape(img)
        
        denoisedimg=cv2.fastNlMeansDenoising(img,None,noiseh,3,11) 
        
        black=np.zeros((h+2*absoffset,w+2*absoffset))
        
    else:
        h,w,c=img.shape
        denoisedimg=cv2.fastNlMeansDenoisingColored(img,None,noiseh,10,3,11) 
        black=np.zeros((h+2*absoffset,w+2*absoffset,3))
    #cv2.imshow('noise',denoisedimg)
    #cv2.waitKey(0)
    
    img=denoisedimg.astype(np.float32)
    k=-offset
    while k<offset+1:
        print(k)
        if direction=='x':
            black[offset+k:h+offset+k,0:w]=black[offset+k:h+offset+k,0:w]+img/(2*offset+1)
            
        if direction=='y':
            black[0:h,offset+k:w+offset+k]=black[offset+k:h+offset+k,0:w]+img/(2*offset+1)
        print(black[0])
        k=k+1
    black=black.astype(np.uint8)
    if direction=='x':
        output=derivget(black,'y',offset)
    if direction=='y':
        output=derivget(black,'x',offset)
    output=output.astype(np.int16)
    output=output*250/np.max(output)
    output=output.astype(np.uint8)
    #cv2.imshow('black',black2)
    #cv2.waitKey(0)
    #cv2.imshow('out',output)
    #cv2.waitKey(0)
    return output
def derivcomb(img,offset):
    absoffset=abs(offset)
    ximg=derivget(img,'x',absoffset)
    yimg=derivget(img,'y',absoffset)
    if len(np.shape(img))==2:
        c=1
        xh,xw=np.shape(ximg)
        yh,yw=np.shape(yimg)
    else:
        xh,xw,xc=ximg.shape
        yh,yw,yc=yimg.shape
    print(xh,xw,yh,yw)
    step=int(absoffset)
    dxtrim=ximg[0:xh-step,0:xw]
    dytrim=yimg[0:yh,0:yw-step]
    # cv2.imshow('dx',dxtrim)
    # cv2.waitKey(0)
    # cv2.imshow('dy',dytrim)
    # cv2.waitKey(0)
    print(dxtrim.shape,dytrim.shape)
    return dytrim+dxtrim
def derivfilt(img,offset):
    lowlim=lderivlim
    highlim=uderivlim
    if len(np.shape(img))==2:
        c=1
        h,w=np.shape(img)
        derivimg=derivcomb(img,offset).astype(np.int16)
        testgray=np.sign(derivimg)*np.sign(graylim-derivimg)
        graypixoutscale=derivimg*np.sign(testgray+abs(testgray))*250/np.max(derivimg)
        print(np.max(graypixoutscale))
        pixoutgray=graypixoutscale.astype(np.uint8)
        return pixoutgray
    else:
        h,w,c=img.shape
        derivimg=derivcomb(img,offset)
        test=np.sign(derivimg-lowlim)*np.sign(highlim-derivimg)
        pixout=derivimg*np.sign(test+abs(test))
        pixoutscale=pixout*250/uderivlim
        pixout2=pixoutscale.astype(np.uint8)
        return pixout2
def run_file_wrapped(filepath):
    tik = time.time()
    filepath1=filepath[0]
    outputloc=filepath[1]
    scanposdict=filepath[2]
    dims=filepath[3]
    try:    
        run_file(filepath1,outputloc,scanposdict,dims)
    except Exception as e:
        print("Exception occurred: ", e)
    tok = time.time()
    print(f"{filepath[0]} - {tok - tik} seconds")

def run_file(img_filepath,outputdir,scanposdict,dims):
    tik = time.time()
    img0 = cv2.imread(img_filepath)
    img=cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    h,w,c=img.shape
    pixcal=1314.08/w #microns/pixel from Leica calibration
    pixcals=[pixcal,876.13/h]
    img_pixels = img.copy().reshape(-1, 3)
    lowlim=np.array([87,100,99])#np.array([108,100,99])#defines lower limit for what code can see as background
    highlim=np.array([140,165,135])#np.array([140,165,135])
    imsmall = cv2.resize(img.copy(), dsize=(256 * k, 171 * k)).reshape(-1,3)
    test=np.sign(imsmall-lowlim)+np.sign(highlim-imsmall)
    pixout=imsmall*np.sign(test+abs(test)) #chooses pixels between provided limits, quickly filtering to potential background pixels
    if len(pixout)==0: #making sure background is identified
        #print('Pixel failed')
        return
    reds=np.bincount(pixout[:,0])
    greens=np.bincount(pixout[:,1])
    blues=np.bincount(pixout[:,2])
    reds[0]=0 #otherwise argmax finds values masked to 0 by pixout
    greens[0]=0
    blues[0]=0
    #print(reds,len(reds))
    reddest=reds.argmax()
    greenest=greens.argmax()
    bluest=blues.argmax()
    backrgb=[reddest,greenest,bluest] #defining background color
    #print(backrgb)
    avg_rgb=rgbfunc(backrgb) #calculates monolayer color based on background color
    #print(backrgb,avg_rgb)
    rgb_pixel_dists = np.sqrt(np.sum((img_pixels - avg_rgb) ** 2, axis=1)) #calculating distance between each pixel and predicted flake RGB
    # rgb_t_count = np.sum(rgb_pixel_dists < t_rgb_dist)
    
    #hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #hsv_pixels = hsv_img.reshape(-1, 3)
    #hue_pixel_dists = np.sqrt((hsv_pixels[:, 0] - flake_color_hsv[0]) ** 2)
    #hue_t_count = np.sum(hue_pixel_dists < t_hue_dist)

    #img_mask = np.logical_and(hue_pixel_dists < t_hue_dist, rgb_pixel_dists < t_rgb_dist)
    img_mask= np.logical_and(rgb_pixel_dists < t_rgb_dist,reddest-img_pixels[:,0]>5) #masking the image to only the pixels close enough to predicted flake color
    img123=cv2.resize(255*img_mask.astype(np.uint8),dsize=(256 * k, 171 * k))
    imfilt=img0*img_mask.reshape((h,w,1))
    imfiltgray=cv2.cvtColor(imfilt, cv2.COLOR_RGB2GRAY)
    imderiv=derivfilt(imfiltgray,3)
    #cv2.imshow('123',cv2.resize(imfilt,dsize=(256 * k, 171 * k)))
    #cv2.waitKey(0)
    #cv2.imshow('123',cv2.resize(imderiv,dsize=(256 * k, 171 * k)))
    #cv2.waitKey(0)
    # Show how many are true, how many are false.
    t_count = np.sum(img_mask)
    # print(t_count)
    print(t_count)
    if t_count < t_color_match_count*len(img_pixels):
         #print('Count failed',t_count)
         return
    print(f"{img_filepath} meets count thresh with {t_count}")
    pixdark=np.sum((img_pixels[:,2]<25)*(img_pixels[:,1]<25)*(img_pixels[:,0]<25))
    if np.sum(pixdark)/len(img_pixels) > 0.1: #edge detection, if more than 10% of the image is too dark, return
        print(f"{img_filepath} was on an edge!")
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
    #print(f"{img_filepath} had {len(label_names)}  dbscan clusters")

    # Thresholding of clusters
    labels = db.labels_
    n_pixels = np.bincount(labels + 1, minlength=len(label_names))
    #print(n_pixels)
    criteria = np.logical_and(n_pixels > t_min_cluster_pixel_count, n_pixels < t_max_cluster_pixel_count)
    h_labels = np.array(label_names)
    #print(h_labels)
    h_labels = h_labels[criteria]
    #print(h_labels)
    h_labels = h_labels[h_labels > 0]

    if len(h_labels) < 1:
        return
    print(f"{img_filepath} had {len(h_labels)} filtered dbscan clusters")

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
    boxes_merged0 = []
    boxes_merged=[]
    eliminated_indexes = []
    eliminated_indexes2 = []
    for _i in range(len(boxes)):
        if _i in eliminated_indexes:
            continue
        i = boxes[_i]
        for _j in range(_i + 1, len(boxes)):
            j = boxes[_j]
            # Ith box is always <= jth box regarding y. Not necessarily w.r.t x.
            # sequence the y layers.
            # just cheat and use Intersection in pixel space method.
            on_i = i.to_mask(dbscan_img)
            on_j = j.to_mask(dbscan_img)

            # Now calculate their intersection. If there's any overlap we'll count that.
            intersection_count = np.logical_and(on_i, on_j).sum()

            if intersection_count > 0:   
                # Extend the first box to include dimensions of the 2nd box.
                x_min = min(i.x, j.x)
                x_max = max(i.x + i.width, j.x + j.width)
                y_min = min(i.y, j.y)
                y_max = max(i.y + i.height, j.y + j.height)
                # print(x_min, x_max)
                # print(y_min, y_max)
                new_width = x_max - x_min
                new_height = y_max - y_min
                i = Box(i.label, x_min, y_min, new_width, new_height)
                eliminated_indexes.append(_j)
        boxes_merged0.append(i)
    for _i in range(len(boxes_merged0)):
        if _i in eliminated_indexes2:
            continue
        i = boxes_merged0[_i]
        for _j in range(_i + 1, len(boxes_merged0)):
            j = boxes_merged0[_j]
            # Ith box is always <= jth box regarding y. Not necessarily w.r.t x.
            # sequence the y layers.
            # just cheat and use Intersection in pixel space method.
            on_i = i.to_mask(dbscan_img)
            on_j = j.to_mask(dbscan_img)

            # Now calculate their intersection. If there's any overlap we'll count that.
            intersection_count = np.logical_and(on_i, on_j).sum()

            if intersection_count > 0:   
                # Extend the first box to include dimensions of the 2nd box.
                x_min = min(i.x, j.x)
                x_max = max(i.x + i.width, j.x + j.width)
                y_min = min(i.y, j.y)
                y_max = max(i.y + i.height, j.y + j.height)
                new_width = x_max - x_min
                new_height = y_max - y_min
                i = Box(i.label, x_min, y_min, new_width, new_height)
                eliminated_indexes2.append(_j)
                #print(_j,' eliminated on second pass')
        boxes_merged.append(i)    
    if not boxes_merged:
        return
    
    # Make patches out of clusters
    wantshape=(int(int(img.shape[1])*scale),int(int(img.shape[0])*scale))
    bscale=wantshape[0]/(256*k) #need to scale up box from dbscan image
    offset=5
    patches = [
        [int((int(b.x) - offset)*bscale), int((int(b.y) - offset)*bscale), int((int(b.width) + 2*offset)*bscale), int((int(b.height) + 2*offset)*bscale)] for b
        in boxes_merged
    ]
    print('patched')
    color=(0,0,255)
    thickness=6
    logger=open(outputdir+"Color Log.txt","a+")
    splits=img_filepath.split("Stage")
    imname=splits[1]
    num=int(os.path.splitext(imname)[0])
    radius=1
    i=-1
    imloc=location(num,dims)
    print(num,'Finding Location')
    while radius>0.1:
        i=i+1
        radius = (int(imloc[0])-int(scanposdict[i][0]))**2+(int(imloc[1])-int(scanposdict[i][2]))**2
    posx=scanposdict[i][1]
    posy=scanposdict[i][3]
    posstr="X:"+str(round(1000*posx,2))+", Y:"+str(round(1000*posy,2))
    img0=cv2.putText(img0, posstr, (100,100), cv2.FONT_HERSHEY_SIMPLEX,3, (0,0,0), 2, cv2.LINE_AA)
    img4=img0.copy()
    for p in patches:
        print(p)
        y_min = int(p[0]+2*offset*bscale/3)
        y_max = int(p[0]+p[2]-2*offset*bscale/3)
        x_min = int(p[1]+2*offset*bscale/3)
        x_max = int(p[1]+p[3]-2*offset*bscale/3) #note that the offsets cancel here
        print(x_min,y_min,x_max,y_max)
        bounds=[max(0,p[1]),min(p[1]+p[3],int(h)),max(0,p[0]),min(p[0]+p[2],int(w))]
        imchunk=img[bounds[0]:bounds[1],bounds[2]:bounds[3]] #identifying bounding box of flake
        xarr=[]
        yarr=[]
        width=round(p[2]*pixcal,1)
        height=round(p[3]*pixcal,1) #microns
        img3=cv2.rectangle(img0,(p[0],p[1]),(p[0]+p[2],p[1]+p[3]),color,thickness) #creating the output images
        img3 = cv2.putText(img3, str(height), (p[0]+p[2]+10,p[1]+int(p[3]/2)), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 2, cv2.LINE_AA)
        img3 = cv2.putText(img3, str(width), (p[0],p[1]-10), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 2, cv2.LINE_AA)
        flakergb=[0,0,0]
        farea=0
        if boundflag==1:
            flakergb,edgeim,farea=edgefind(imchunk,avg_rgb,pixcals) #calculating border pixels
            print('Edge found')
            img4=cv2.rectangle(img4,(p[0],p[1]),(p[0]+p[2],p[1]+p[3]),color,thickness)
            img4 = cv2.putText(img4, str(height), (p[0]+p[2]+10,p[1]+int(p[3]/2)), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 2, cv2.LINE_AA)
            img4 = cv2.putText(img4, str(width), (p[0],p[1]-10), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 2, cv2.LINE_AA)
            print(edgeim.shape)
            img4[bounds[0]:bounds[1],bounds[2]:bounds[3]]=img4[bounds[0]:bounds[1],bounds[2]:bounds[3]]+edgeim
        logstr=str(num)+','+str(farea)+','+str(flakergb[0])+','+str(flakergb[1])+','+str(flakergb[2])+','+str(backrgb[0])+','+str(backrgb[1])+','+str(backrgb[2])
        logger.write(logstr+'\n')
    logger.close()

    cv2.imwrite(os.path.join(outputdir, os.path.basename(img_filepath)),img3)
    if boundflag==1:
        cv2.imwrite(os.path.join(outputdir+"\\AreaSort\\", str(int(farea))+'_'+os.path.basename(img_filepath)),img4)
    
    tok = time.time()
    print(f"{img_filepath} - {tok - tik} seconds")
def edgefind(imchunk,avg_rgb,pixcals): #this identifies the edges of flakes, resource-intensive but useful for determining if flake ID is working
    pixcalw=pixcals[0]
    pixcalh=pixcals[1]
    edgerad=20
    imchunk2=imchunk.copy()
    impix=imchunk.copy().reshape(-1, 3)
    dims=np.shape(imchunk)
    flakeid=np.sqrt(np.sum((impix - avg_rgb) ** 2, axis=1))<t_rgb_dist #a mask for pixel color
    maskpic=np.reshape(flakeid,(dims[0],dims[1],1))
    freds=np.bincount(impix[:,0]*flakeid)
    fgreens=np.bincount(impix[:,1]*flakeid)
    fblues=np.bincount(impix[:,2]*flakeid)
    freds[0]=0 #otherwise argmax finds values masked to 0 by flakeid
    fgreens[0]=0
    fblues[0]=0
    freddest=freds.argmax() #determines flake RGB as the most common R,G,B value in identified flake region
    fgreenest=fgreens.argmax()
    fbluest=fblues.argmax() 
    rgb=[freddest,fgreenest,fbluest]
    h,w,c=imchunk.shape
    flakeid2=np.sqrt(np.sum((impix - rgb) ** 2, axis=1))<5 #a mask for pixel color
    maskpic2=np.reshape(flakeid2,(dims[0],dims[1],1))
    indices = np.argwhere(np.any(maskpic2 > 0, axis=2)) #flake region
    farea=round(len(indices)*pixcalw*pixcalh,1)
    grayimg=cv2.cvtColor(imchunk, cv2.COLOR_BGR2GRAY)
    grayimg=cv2.fastNlMeansDenoising(grayimg,None,2,3,11) 
    edgeim=np.reshape(cv2.Canny(grayimg,5,15),(h,w,1))
    edgeim=edgeim.astype(np.int16)*np.array([25,25,25])/255
    edgeim=edgeim.astype(np.uint8)
    return rgb,edgeim,farea
def rgbfunc(rgbarr): #outputs flake color based on input background color, values determined empirically
    red=rgbarr[0]
    green=rgbarr[1]
    blue=rgbarr[2]
    rval=int(round(0.8643*red-2.55,0))
    gval=int(round(0.8601*green+9.6765,0))
    bval=blue+4
    #print('coloring')
    return np.array([rval,gval,bval])
def dimget(inputdir): #finds scan size from microscope file
        filename=inputdir+"/leicametadata/TileScan_001.xlif"
        try:
            with open(filename,'r') as file:
                rawdata=file.read()
            rawdata2=rawdata.partition('</Attachment>')
            rawdata=rawdata2[0]
            size=len(rawdata)
            xarr=[]
            yarr=[]
            while size>10:
                rawdata2=rawdata.partition('FieldX="')
                rawdata=rawdata2[2]
                rawdata3=rawdata.partition('" FieldY="')
                xd=int(rawdata3[0])
                rawdata=rawdata3[2]
                rawdata4=rawdata.partition('" PosX="')
                yd=int(rawdata4[0])
                rawdata=rawdata4[2]
                rawdata5=rawdata.partition('" />')
                rawdata=rawdata5[2]
                xarr.append(xd)
                yarr.append(yd)
                size=len(rawdata)
                #print(size)
            xarr=np.array(xarr)
            yarr=np.array(yarr)
            print('Your scan is '+str(np.max(xarr)+1)+' by '+str(np.max(yarr)+1))
            return np.max(xarr)+1,np.max(yarr)+1
        except:
            return 1,1
def posget(inputdir): #finds image location from microscope file
    filename=inputdir+"/leicametadata/TileScan_001.xlif"
    print(filename)
    posarr=[]
    with open(filename,'r') as file:
        rawdata=file.read()
        rawdata2=rawdata.partition('</Attachment>')
        rawdata=rawdata2[0]
        size=len(rawdata)
        while size>10:
                rawdata2=rawdata.partition('FieldX="')
                rawdata=rawdata2[2]
                rawdata3=rawdata.partition('" FieldY="')
                xd=int(rawdata3[0])
                rawdata=rawdata3[2]
                rawdata4=rawdata.partition('" PosX="')
                yd=int(rawdata4[0])
                rawdata5=rawdata4[2]
                rawdata6=rawdata5.partition('" PosY="')
                posx=float(rawdata6[0])
                rawdata7=rawdata6[2]
                rawdata8=rawdata7.partition('" PosZ="')
                posy=float(rawdata8[0])
                rawdata9=rawdata8[2]
                rawdata10=rawdata9.partition('" />')
                rawdata=rawdata10[2]
                posarr.append([xd,posx,yd,posy])
                size=len(rawdata)
                #print(size)
        posarr=np.array(posarr)
        print('Position dict made')
        return posarr
def location(m,dimset):
    outset=dimset
    height=int(outset[1])
    width=int(outset[0])
    row =m % height
    column = (m-row)/height
    print(m,column,row)
    return column,row,height-1,width-1
def plotmaker(mlist,dims,directory):
    imx=1314.09/1000
    imy=875.89/1000 #mm
    #parr=[]
    plt.figure(figsize=(18,18))
    print(mlist)
    for m in mlist:
        x,y,maxy,maxx=location(m,dims)
        print(x,y,maxy,maxx)
        plt.scatter(x*imx,y*imy)
        plt.text(x*imx, y*imy+.03, m, fontsize=9)
        #parr.append([m,round(x*imx,1),round(y*imy,1)])
    boundx=[0,maxx*imx,maxx*imx,0,0]
    boundy=[0,0,maxy*imy,maxy*imy,0]
    plt.plot(boundx,boundy)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
    plt.savefig(directory+"coordmap.jpg")
    plt.close()
def main(args):
    inputfile=args.q
    file1=open(str(inputfile))
    inputs=file1.readlines()
    cleaninputs=[]
    for line in inputs:
        line=line.strip("\n")
        slicer=line.find("OutputDir:")
        inputdir=line[10:slicer-2] #starts after the length of "InputDir: "
        slicer2=slicer+11
        outputdir=line[slicer2:]
        cleaninputs.append([inputdir,outputdir])
    print(cleaninputs)
    for pair in cleaninputs:
        input_dir=pair[0]
        outputloc=pair[1]
        os.makedirs(outputloc, exist_ok=True)
        os.makedirs(outputloc+"\\AreaSort\\", exist_ok=True)
        files = glob.glob(os.path.join(input_dir, "*"))
        
        files = [f for f in files if "Stage" in f]
        files.sort(key=len)
        # Filter files to only have images.
        #smuggling outputloc into pool.map by packaging it with the iterable, gets unpacked by run_file_wrapped
        dims=dimget(input_dir)
        n_proc = os.cpu_count()-threadsave #config.jobs if config.jobs > 0 else 
        logger=open(outputloc+"Color Log.txt","w+")
        logger.write('N,A,Rf,Gf,Bf,Rw,Gw,Bw\n')
        logger.close()
        tik = time.time()
        scanposdict=posget(input_dir)
        files = [[f,outputloc,scanposdict,dims] for f in files if os.path.splitext(f)[1] in [".jpg", ".png", ".jpeg"]]
        with Pool(n_proc) as pool:
            pool.map(run_file_wrapped, files)
        tok = time.time()
        filecounter = glob.glob(os.path.join(outputloc, "*"))
        filecounter=[f for f in filecounter if os.path.splitext(f)[1] in [".jpg", ".png", ".jpeg"]]
        filecounter2=[f for f in filecounter if "Stage" in f]
        #print(filecounter2)
        #print(filecounter2)
        filecount=len(filecounter2)
        f=open(outputloc+"Summary.txt","a+")
        f.write(f"Total for {len(files)} files: {tok - tik} = avg of {(tok - tik) / len(files)} per file on {n_proc} logical processors\n")
        f.write(str(filecount)+' identified flakes\n')
        
        f.write('flake_colors_rgb='+str(flake_colors_rgb)+'\n')
        f.write('t_rgb_dist='+str(t_rgb_dist)+'\n')
        #f.write('t_hue_dist='+str(t_hue_dist)+'\n')
        f.write('t_red_dist='+str(t_red_dist)+'\n')
        #f.write('t_red_cutoff='+str(t_red_cutoff)+'\n')
        f.write('t_color_match_count='+str(t_color_match_count)+'\n')
        f.write('t_min_cluster_pixel_count='+str(t_min_cluster_pixel_count)+'\n')
        f.write('t_max_cluster_pixel_count='+str(t_max_cluster_pixel_count)+'\n')
        f.write('k='+str(k)+"\n\n")
        f.close()
        flist=open(outputloc+"Imlist.txt","w+")
        flist.write("List of Stage Numbers for copying to Analysis Sheet"+"\n")
        flist.close()
        flist=open(outputloc+"Imlist.txt","a+")
        fwrite=open(outputloc+"By Area.txt","w+")
        fwrite.write("Num, A"+"\n")
        fwrite.close()
        fwrite=open(outputloc+"By Area.txt","a+")
        numlist=[]
        for file in filecounter2:
            splits=file.split("Stage")
            num=splits[1]
            number=os.path.splitext(num)[0]
            numlist.append(int(number))
        numlist=np.sort(np.array(numlist))
        for number in numlist:
            flist.write(str(number)+"\n")
        plotmaker(numlist,dims,outputloc) #creating cartoon for file
        flist.close()
        #print(outputloc+"Color Log.txt")
        N,A,Rf,Gf,Bf,Rw,Gw,Bw=np.loadtxt(outputloc+"Color Log.txt", skiprows=1,delimiter=',',unpack=True)
        pairs=[]
        i=0
        while i<len(A):
            pair=np.array([N[i],A[i]])
            pairs.append(pair)
            i=i+1
        #print(pairs)
        pairsort=sorted(pairs, key = lambda x : x[1],reverse=True)
        #print(pairs,pairsort)
        for pair in pairsort:
            writestr=str(int(pair[0]))+', '+str(pair[1])+'\n'
            fwrite.write(writestr)
        fwrite.close()
        
        print(f"Total for {len(files)} files: {tok - tik} = avg of {(tok - tik) / len(files)} per file")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find graphene flakes on SiO2. Currently configured only for "
                                                 "exfoliator dataset")
    parser.add_argument("--q", required=True, type=str,
                        help="Directory containing images to process. Optional unless running in headless mode")
    args = parser.parse_args()
    main(args)