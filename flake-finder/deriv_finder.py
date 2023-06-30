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
threadsave=8
k=4
uderivlim=np.array([5,7,30]) #bgr
lderivlim=np.array([0,0,0])
graylim=5
offset=3
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
#core functions
def run_file_wrapped(filepath):
    tik = time.time()
    filepath1=filepath[0]
    outputloc=filepath[1]
    #scanposdict=filepath[2]
    dims=filepath[2]
    try:    
        run_file(filepath1,outputloc,dims)
    except Exception as e:
        print("Exception occurred: ", e)
    tok = time.time()
    print(f"{filepath[0]} - {tok - tik} seconds")
def run_file(img_filepath,outputloc,dims):
    tik = time.time()
    img0 = cv2.imread(img_filepath)
    img=img0
    #img=cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    h,w,c=img.shape
    pixcal=1314.08/w #microns/pixel from Leica calibration
    pixcals=[pixcal,876.13/h]
    img_pixels = img.copy().reshape(-1, 3)
    imsmall0 = cv2.resize(img.copy(), dsize=(256 * k, 171 * k))
    imsmall=imsmall0.reshape(-1,3)
    #imsmall0=cv2.cvtColor(imsmall0, cv2.COLOR_BGR2GRAY)
    
    print(img_filepath)
    deriv=colorgraycomp(imsmall0,offset)#derivfilt(imsmall0,offset)#derivget(imsmall0,'x',1)
    cv2.imshow('deriv',deriv)
    cv2.waitKey(0)
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
        #scanposdict=posget(input_dir)
        files = [[f,outputloc,dims] for f in files if os.path.splitext(f)[1] in [".jpg", ".png", ".jpeg"]]
        #with Pool(n_proc) as pool:
            #pool.map(run_file_wrapped, files)
        for file in files:
            run_file_wrapped(file)
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
        
        # f.write('flake_colors_rgb='+str(flake_colors_rgb)+'\n')
        # f.write('t_rgb_dist='+str(t_rgb_dist)+'\n')
        # #f.write('t_hue_dist='+str(t_hue_dist)+'\n')
        # f.write('t_red_dist='+str(t_red_dist)+'\n')
        # #f.write('t_red_cutoff='+str(t_red_cutoff)+'\n')
        # f.write('t_color_match_count='+str(t_color_match_count)+'\n')
        # f.write('t_min_cluster_pixel_count='+str(t_min_cluster_pixel_count)+'\n')
        # f.write('t_max_cluster_pixel_count='+str(t_max_cluster_pixel_count)+'\n')
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
        # plotmaker(numlist,dims,outputloc) #creating cartoon for file
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
#helper functions
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
    cv2.imshow('out',output)
    cv2.waitKey(0)
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
        graypixoutscale=derivimg*np.sign(testgray+abs(testgray))*250/graylim
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
def contourfiller(img):
    ret, thresh = cv2.threshold(img, 5, 15, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        cv2.drawContours(img,[c],0,(255,255,255),-1)
    return img
def cwrap(img,k):
    i=0
    while i<k:
        img=contourfiller(img)
        i=i+1
    return img
def colorgraycomp(img,offset):
    grayimg=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayimg2=cv2.fastNlMeansDenoising(grayimg,None,2,3,11) 
    sobelx = cv2.Canny(grayimg2,5,15)
    cv2.imshow('edgefind',sobelx)
    cv2.waitKey(0)
    sobelx2=cwrap(sobelx,3)
    
    cv2.imshow('edgefind2',sobelx2)
    cv2.waitKey(0)
    grayderiv=derivfilt(grayimg,offset)
    #grayd2=derivget2(grayimg,'x',offset)
    longset=offset*5
    grayderivlong=derivfilt(grayimg,longset)
    grayderivlong=cwrap(grayderivlong,2)
    grayderiv=cwrap(grayderiv,2)
    h,w=np.shape(grayderiv)
    grayderiv=grayderiv[2*offset:h-2*offset,2*offset:w-2*offset]
    #grayderivlong=grayderivlong[]
    cderiv=derivfilt(img,offset)
    cderivb=cderiv[:,:,0].astype(np.int16)
    print(cderivb)
    testblue=1-np.sign(grayderivlong)
    compim=testblue*grayderiv #filters edges with no blue information
    #cv2.imshow('long',grayderivlong)
    #cv2.waitKey(0)
    #cv2.imshow('short',grayderiv)
    #cv2.waitKey(0)
    return compim.astype(np.uint8)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find graphene flakes on SiO2. Currently configured only for "
                                                 "exfoliator dataset")
    parser.add_argument("--q", required=True, type=str,
                        help="Directory containing images to process. Optional unless running in headless mode")
    args = parser.parse_args()
    main(args)