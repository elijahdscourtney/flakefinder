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
from scipy.spatial import distance_matrix
import scipy.signal as sig



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

    def to_mask(self, img, b=-6):
        h, w, c = img.shape
        boundx=min(0,self.x)
        boundy=min(0,self.y)
        return np.logical_and.outer(
            np.logical_and(np.arange(boundy,h) >= self.y - b, np.arange(boundy,h) <= self.y + self.height + b),
            np.logical_and(np.arange(boundx,w) >= self.x - b, np.arange(boundx,w) <= self.x + self.width + b),
        )

flake_colors_rgb = [
    [0,0,0],
]

output_dir = 'cv_output'
threadsave=8 #number of threads NOT allocated when running
boundflag=1
t_rgb_dist = 50
t_gray_dists=[5,12]
t_red_dist = 12
t_red_cutoff = 0.1 #fraction of the chunked image that must be more blue than red to be binned
t_color_match_count = 2*0.000225 #fraction of image that must look like monolayers
k = 2
t_min_cluster_pixel_count = 30*(k/4)**2  # flake too small
t_max_cluster_pixel_count = 20000*(k/4)**2  # flake too large
cutoff=200 #um2
 # scale factor for DB scan. recommended values are 3 or 4. Trade-off in time vs accuracy. Impact epsilon.
scale=1 #the resolution images are saved at, relative to the original file. Does not affect DB scan

# This would be a decorator but apparently multiprocessing lib doesn't know how to serialize it.
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
def graytest(imgray,graypix,peaks,t_gray_dists):
    img_mask2=[]
    peak=0
    i=0
    
    while i<len(peaks):
        peak=peaks[i]
        #print(peak)
        
        img_mask=np.logical_and(imgray < peak-t_gray_dists[0],imgray>peak-t_gray_dists[1])
        h,w=img_mask.shape
        #print(np.sum(img_mask))
        if np.sum(img_mask)<0.05*len(graypix):
            img_mask2=255*img_mask.reshape((h,w,1))
            img_mask2=img_mask2.astype(np.uint8)
            #print(img_mask2.shape,type(img_mask2))
            #img_mask2=cv2.resize(img_mask2, dsize=(256 * k, 171 * k))
            break
        else:
            i=i+1
    return np.array(img_mask2), peak
def graychunktest(imgray,graypix,peak,t_gray_dists):
    img_mask2=[]
    img_mask=np.logical_and(imgray < peak-t_gray_dists[0],imgray>peak-t_gray_dists[1])
    h,w=img_mask.shape
    img_mask2=img_mask.reshape((h,w,1)).astype(np.uint8)
    return np.array(img_mask2)
def bgtest(imsmall,imgray,peak):
    imgray=imsmall
    bgmask=np.logical_and(imgray < peak+3,imgray>peak-3)
    background=imsmall*bgmask
    impix=background.reshape((-1,3))
    freds=np.bincount(impix[:,2])
    fgreens=np.bincount(impix[:,1])
    fblues=np.bincount(impix[:,0])
    freds[0]=0 #otherwise argmax finds values masked to 0 by flakeid
    fgreens[0]=0
    fblues[0]=0
    freddest=freds.argmax() #determines flake RGB as the most common R,G,B value in identified flake region
    fgreenest=fgreens.argmax()
    fbluest=fblues.argmax() 
    backrgb=[freddest,fgreenest,fbluest]
    return backrgb
def run_file(img_filepath,outputdir,scanposdict,dims):
    tik = time.time()
    
    img0 = cv2.imread(img_filepath)
    imgray0=cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
    imgray=cv2.resize(imgray0, dsize=(256 * k, 171 * k))
    imgray=cv2.fastNlMeansDenoising(imgray,None,2,3,11) 
    graypix=imgray.copy().reshape(-1, 3)
    graycount,edges=np.histogram(graypix,bins=256,range=(0,255))
    peaks,props=sig.find_peaks(graycount,height=0.01*len(graypix))
    img_mask2,peak=graytest(imgray,graypix,peaks,t_gray_dists)
    imsmall=cv2.resize(img0, dsize=(256 * k, 171 * k))
    backrgb=bgtest(imsmall,imgray,peak)
    colormask=np.sqrt(np.sum((imsmall - backrgb) ** 2, axis=2))<t_rgb_dist
    if len(img_mask2)==0 or peak<50:
        #print('no good pixels')
        return
    h,w,c=img0.shape
    #print(colormask.shape)
    pixcal=1314.08/w #microns/pixel from Leica calibration
    pixcals=[pixcal,876.13/h]
    img_pixels = img0.copy().reshape(-1, 3)
    
    #img_mask2=cv2.fastNlMeansDenoising(img_mask2,None,50,15,33)>60 
    t_count = np.sum(img_mask2)
    # print(t_count)
    if t_count < t_color_match_count*len(graypix):
         print('Count failed',t_count)
         return 
    #pixdark=np.sum((img_pixels[:,2]<25)*(img_pixels[:,1]<25)*(img_pixels[:,0]<25))
    pixdark=np.sum(graypix<50)
    if np.sum(pixdark)/len(img_pixels) > 0.1: #edge detection, if more than 10% of the image is too dark, return
        print(f"{img_filepath} was on an edge!")
        return
    print(f"{img_filepath} meets count thresh with {t_count}")
    
    
    img_mask2=img_mask2.reshape((171*k,256*k,1))
    # cv2.imshow('gimg',img_mask2)
    # cv2.waitKey(0)
    colormask=colormask.reshape((171*k,256*k,1)).astype(np.uint8)
    img_mask2=img_mask2*colormask
    # cv2.imshow('gcimg',img_mask2)
    # cv2.waitKey(0)
    #print(img_mask2.shape)
    # Create Masked image
    
    # DB SCAN
    
    dbscan_img = img_mask2
    # db = DBSCAN(eps=2.0, min_samples=6, metric='euclidean', algorithm='auto', n_jobs=-1)
    db = DBSCAN(eps=3, min_samples=6, metric='euclidean', algorithm='auto', n_jobs=1)
    
    indices = np.dstack(np.indices(dbscan_img.shape[:2]))
    xycolors = np.concatenate((dbscan_img, indices), axis=-1)
    
    feature_image = np.reshape(xycolors, [-1, 3])
    
    tik2=time.time()
    db.fit(feature_image)
    tok2=time.time()
    label_names = range(-1, db.labels_.max() + 1)
    print(f"{img_filepath} had {len(label_names)}  dbscan clusters")
    
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
        #print('no filtered clusters')
        return
    print(f"{img_filepath} had {len(h_labels)} filtered dbscan clusters")
    print('peak',peak)
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
    # Make patches
    wantshape=(int(int(img0.shape[1])*scale),int(int(img0.shape[0])*scale))
    bscale=wantshape[0]/(256*k) #need to scale up box from dbscan image
    offset=5
    patches = [
        [int((int(b.x) - offset)*bscale), int((int(b.y) - offset)*bscale), int((int(b.width) + 2*offset)*bscale), int((int(b.height) + 2*offset)*bscale)] for b
        in boxes_merged
    ]
    print('patched')
    color=(0,0,255)
    color2=(0,255,0)
    thickness=6
    logger=open(outputdir+"Color Log.txt","a+")
    poscount=1#0
    try:
        splits=img_filepath.split("Stage")
        imname=splits[1]
        num=int(os.path.splitext(imname)[0])
    except:
        splits=img_filepath.split("\\")
        imname=splits[len(splits)-1]
        num=int(os.path.splitext(imname)[0])
    radius=1
    i=-1
    imloc=location(num,dims)
    xd=imloc[0]
    yd=imloc[1]
    print(num,'Finding Location')
    try:
        # while radius>0.1:
        #     i=i+1
        #     radius = (int(imloc[0])-int(scanposdict[i][0]))**2+(int(imloc[1])-int(scanposdict[i][2]))**2
        impos=scanposdict[int(yd),int(xd)]
        #print(impos)
        posx=impos[0]
        posy=impos[1]
        posstr="X:"+str(round(1000*posx,2))+", Y:"+str(round(1000*posy,2))
    except:
        print('pos failed')
        posstr=""
    
    #print('bg',backrgb)
    img0=cv2.putText(img0, posstr, (100,100), cv2.FONT_HERSHEY_SIMPLEX,3, (0,0,0), 2, cv2.LINE_AA)
    img4=img0.copy()
    h,w,c=img0.shape
    patches2=[p for p in patches if boundtest(boundmaker(p,h,w),h,w)]
    fareamax=0
    if len(patches2)>0:
        for p in patches2:
            #print(p)
            bounds=boundmaker(p,h,w)
            imchunkgray=imgray0[bounds[0]:bounds[1],bounds[2]:bounds[3]]
            imchunk=img0[bounds[0]:bounds[1],bounds[2]:bounds[3]] #identifying bounding box of flake
            imchunkmask=graychunktest(imchunkgray,imchunkgray.reshape(-1,1),peak,t_gray_dists)
                #print(imchunkmask.shape)
                #print('masked')
            width=round(p[2]*pixcal,1)
            height=round(p[3]*pixcal,1) #microns
            flakergb=[0,0,0]
            flakergb,edgeim,farea=edgefind(imchunk,imchunkmask,pixcals) #calculating border pixels
            if farea>cutoff:
                img3=cv2.rectangle(img0,(p[0],p[1]),(p[0]+p[2],p[1]+p[3]),color,thickness) #creating the output images
                img3 = cv2.putText(img3, str(height), (p[0]+p[2]+10,p[1]+int(p[3]/2)), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 2, cv2.LINE_AA)
                img3 = cv2.putText(img3, str(width), (p[0],p[1]-10), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 2, cv2.LINE_AA)
                img3 = cv2.putText(img3, str(10*int(farea/10)), (p[0],p[1]+p[3]+35), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 2, cv2.LINE_AA)
                if farea>fareamax:
                    fareamax=farea
                if boundflag==1:
                    print('Edge found')
                    
                    img4=cv2.rectangle(img4,(p[0],p[1]),(p[0]+p[2],p[1]+p[3]),color,thickness)
                    img4 = cv2.putText(img4, str(height), (p[0]+p[2]+10,p[1]+int(p[3]/2)), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 2, cv2.LINE_AA)
                    img4 = cv2.putText(img4, str(width), (p[0],p[1]-10), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 2, cv2.LINE_AA)
                    img4 = cv2.putText(img4, str(10*int(farea/10)), (p[0],p[1]+p[3]+35), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,0), 2, cv2.LINE_AA)
                    img4[bounds[0]:bounds[1],bounds[2]:bounds[3]]=img4[bounds[0]:bounds[1],bounds[2]:bounds[3]]+edgeim
                try:
                    logstr=str(num)+','+str(farea)+','+str(flakergb[0])+','+str(flakergb[1])+','+str(flakergb[2])+','+str(backrgb[0])+','+str(backrgb[1])+','+str(backrgb[2])
                except:
                    logstr=""
                logger.write(logstr+'\n')
        logger.close()
        if fareamax>cutoff:
            cv2.imwrite(os.path.join(outputdir, os.path.basename(img_filepath)),img3)
            if boundflag==1:
                cv2.imwrite(os.path.join(outputdir+"\\AreaSort\\", str(int(fareamax))+'_'+os.path.basename(img_filepath)),img4)
    
    tok = time.time()
    print(f"{img_filepath} - {tok - tik} seconds")
def boundtest(bounds,h,w): #y1,y2,x1,x2
    y1,y2,x1,x2=bounds
    #print(bounds,h,w)
    delt=0.05
    if y2>delt*h and x2>delt*w and y1<(1-delt)*h and x1<(1-delt)*w:
        return 1
    else:
        print(y1,(1-delt)*h,y2,delt*h,x1,(1-delt)*w,x2,delt*w)
        return 0
def edgefind(imchunk,imchunkmask,pixcals): #this identifies the edges of flakes, resource-intensive but useful for determining if flake ID is working
    pixcalw=pixcals[0]
    pixcalh=pixcals[1]
    #print('masking')
    maskedpic=imchunk*imchunkmask
    impix=maskedpic.reshape(-1, 3)
    freds=np.bincount(impix[:,0])
    fgreens=np.bincount(impix[:,1])
    fblues=np.bincount(impix[:,2])
    freds[0]=0 #otherwise argmax finds values masked to 0 by flakeid
    fgreens[0]=0
    fblues[0]=0
    freddest=freds.argmax() #determines flake RGB as the most common R,G,B value in identified flake region
    fgreenest=fgreens.argmax()
    fbluest=fblues.argmax() 
    rgb=[freddest,fgreenest,fbluest]
    h,w,c=imchunk.shape
    farea=round(np.sum(imchunkmask)*pixcalw*pixcalh,1)
    grayimg=cv2.cvtColor(imchunk, cv2.COLOR_BGR2GRAY)
    grayimg=cv2.fastNlMeansDenoising(grayimg,None,2,3,11) 
    edgeim=np.reshape(cv2.Canny(grayimg,5,15),(h,w,1))
    edgeim=edgeim.astype(np.int16)*np.array([25,25,25])/255
    edgeim=edgeim.astype(np.uint8)
    return rgb,edgeim,farea
def boundmaker(p,h,w):
    return [max(0,p[1]),min(p[1]+p[3],int(h)),max(0,p[0]),min(p[0]+p[2],int(w))]
def dimget(inputdir):
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
def posget(inputdir):
    filename=inputdir+"/leicametadata/TileScan_001.xlif"
    print(filename)
    posarr=np.zeros((100,100,2))
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
                posarr[yd,xd]=np.array([posy,posx])
                #print(yd,xd,posarr[yd,xd])
                #posarr.append([xd,posx,yd,posy])
                size=len(rawdata)
                #print(size)
        #posarr=np.array(posarr)
        
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
    parr=[]
    plt.figure(figsize=(18,18))
    print(mlist)
    for m in mlist:
        x,y,maxy,maxx=location(m,dims)
        print(x,y,maxy,maxx)
        plt.scatter(x*imx,y*imy)
        plt.text(x*imx, y*imy+.03, m, fontsize=9)
        parr.append([m,round(x*imx,1),round(y*imy,1)])
    boundx=[0,maxx*imx,maxx*imx,0,0]
    boundy=[0,0,maxy*imy,maxy*imy,0]
    plt.plot(boundx,boundy)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
    plt.savefig(directory+"coordmap.jpg")
    plt.close()
    return parr
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
        
        
        #files = [f for f in files if "Stage" in f]
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
        #for file in files:
            #run_file_wrapped(file)
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
        f.write('t_red_cutoff='+str(t_red_cutoff)+'\n')
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
        parr=plotmaker(numlist,dims,outputloc)
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