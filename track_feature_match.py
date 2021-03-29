#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 13:12:05 2020

@author : cloudTomat
@author : samdav / astrosurf.com

"""

SIDERAL_SPEED=15.041

import numpy as np 
#from scipy import stats
import cv2 
import math as math
from os import listdir
from os.path import isfile, join
import exifread
from datetime import datetime
#from matplotlib import pyplot as plt
from astropy.io import fits

debug=False
if debug:
    import matplotlib.pyplot as plt


path="/mnt/data/sandbox/test_image/fits/"

def GetExifDate(imagePath):
    with open(imagePath, 'rb') as fh:
        print(imagePath)
        tags = exifread.process_file(fh, stop_tag="EXIF DateTimeOriginal")
        dateTaken = tags["EXIF DateTimeOriginal"]
        date_obj = datetime.strptime(str(dateTaken), '%Y:%m:%d %H:%M:%S')
        return date_obj
    
def GetFitsDate(imagePath):
    with open(imagePath, 'rb') as fh:
        print(imagePath)
        hdul = fits.open(imagePath)  # open a FITS file
        tags = hdul[0].header  # the primary HDU header
        dateTaken = tags["DATE-OBS"]
        print (dateTaken)
        format = '%Y-%m-%dT%H:%M:%S.%f'

        try:
            # ASI ZWO compliant
            #example DATE-OBS 2021-02-12T19:59:41.106
            date_obj = datetime.strptime(str(dateTaken),format)
        except:

            try :
                # QHY compliant
                #example
                # DATE-OBS 2021-02-12
                # TIME-OBS 19:59:41
                dateTaken = tags["DATE-OBS"]
                timeTaken = tags["TIME-OBS"]
                date_obj = datetime.strptime(str(dateTaken)+"T"+str(timeTaken)+".0", '%Y-%m-%dT%H:%M:%S.%f')
            except :
                try:
                    # ATIK314/starlight/SBIG compliant
                    #example DATE-OBS 2021-02-12T19:59:41
                    date_obj = datetime.strptime(str(dateTaken), '%Y-%m-%dT%H:%M:%S')
                except:
                    print ("Error while reading date and time obs in FITS HEADER")
                    print ("Erreur lecture FITS HEADER - DATE and TIME OBSf")
        return date_obj




def RemoveOutliers(results,m):
    dist=results[:,1]
    idx = np.where(abs(dist - np.mean(dist)) <= m * np.std(dist))[0]
    results=results[idx]

    angle=results[:,2]
    idx = np.where(abs(angle - np.mean(angle)) <= m * np.std(angle))
    results=results[idx]
    return results


def crop(frame,sizeX,sizeY,offsetX=0,offsetY=0):
    w = frame.shape[1] 
    h = frame.shape[0]
    start=(int(w/2-sizeX/2)-offsetX,int(h/2-sizeY/2)-offsetY)
    end=(int(w/2+sizeX/2)-offsetX,int(h/2+sizeY/2)-offsetY) 
    frameCrop=frame[start[1]:end[1],start[0]:end[0]]
    return frameCrop


def calcMove_old(im1Path,im2Path):
    if debug:
       fig = plt.figure(figsize=(8, 3))
       ax1 = plt.subplot(1, 3, 1)
       ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
       ax3 = plt.subplot(1, 3, 3)
    
    im1Read = cv2.imread(im1Path) 
    im2Read = cv2.imread(im2Path)
    
    im1=crop(im1Read,800,800)
    im2=crop(im2Read,800,800)
    
    im1=im1.astype(np.float32)
    im2=im2.astype(np.float32)
    
    
    mapper = cv2.reg_MapperGradShift()
    mappPyr = cv2.reg_MapperPyramid(mapper)
    
    resMap = mappPyr.calculate(im1, im2)
    mapShift = cv2.reg.MapTypeCaster_toShift(resMap)
    solution=mapShift.getShift()
    shift=[solution[0][0],solution[1][0]]
    
    if debug    :  
        ax1.imshow(im1Read, cmap='gray')
        ax1.set_axis_off()
        ax1.set_title('Reference image')
        
        ax2.imshow(im2Read.real, cmap='gray')
        ax2.set_axis_off()
        ax2.set_title('Offset image')
        plt.show()        
    
    norm=np.linalg.norm(shift)
    angle=math.atan2(shift[1], shift[0])*180./math.pi
    
    return (norm,angle)

def calcMove(im1Path,im2Path):
    format = "DLSR"

    if ".fit" in im1Path or ".FIT" in im1Path:
        format = "FITS_MONO"

        # convert to uint 8
        hdul = fits.open(im1Path)
        ref = cv2.normalize(src=hdul[0].data, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        hdul = fits.open(im2Path)
        new = cv2.normalize(src=hdul[0].data, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    else:
        ref = cv2.imread(im1Path) 
        new = cv2.imread(im2Path)  
    


    offset=60

    #Prepare the Images
    #------------------

    
    
    #Scale the image to increase accuracy
    factor=1
    if factor!=1:
        width = int(ref.shape[1] * factor)
        height = int(ref.shape[0] * factor)
        ref=cv2.resize(ref,(width, height))
        new=cv2.resize(new, (width, height))

    # Convert to grayscale 
    refBw = cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY) 
    newBw = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY) 
    
    #crop New img center, 10% of the image
    w = new.shape[1] 
    h = new.shape[0]
    startNew=(int(w/2-w*0.1),int(h/2-h*0.1))
    endNew=(int(w/2+w*0.1),int(h/2+h*0.1)) 
    newBw=newBw[startNew[1]:endNew[1],startNew[0]:endNew[0]]

    #reduce ref image to only search in a square  around image center
    #reduce processing time but limit max possible move
    startRef=(int(w/2-w*0.1-offset),int(h/2-h*0.1-offset))
    endRef=(int(w/2+w*0.1)+offset,int(h/2+h*0.1)+offset) 
    refBw=refBw[startRef[1]:endRef[1],startRef[0]:endRef[0]]

    #Pattern Matching Algo
    #----------------------
    #Select pattern match method 
    method = cv2.TM_CCOEFF_NORMED
    #find best match of new img crop in ref 
    res = cv2.matchTemplate(refBw,newBw,method)
    #Top left coordinate of best match
    bestMatch=np.asarray(np.unravel_index(res.argmax(), res.shape))

    #print(bestMatch)

    #Central pixel is best match with pixel resolution
    dstX0=float(bestMatch[0]-offset)/factor
    dstY0=float(bestMatch[1]-offset)/factor

    #now working on subpixel resolution algorithm
    #---------------------------------
    #isolate neighbour matrix of best result in each direction
    neig=np.copy(res[bestMatch[0]-1:bestMatch[0]+2,bestMatch[1]-1:bestMatch[1]+2])
    neig=neig.transpose() #don't know exactly why X/Y are reverted 
    
    #Affine transformation of neighbour matrix
    #Min becomes 0
    #Max unmodified
    maxVal=np.max(neig)
    minVal=np.min(neig)
    a=maxVal/(maxVal-minVal)
    b=-maxVal*minVal/(maxVal-minVal)
    neig=a*neig+b

    #print(neig)

    #New max and new min after affine transform
    maxVal=np.max(neig)     #not necessary (just in case affine transform is modified)
    minVal=np.min(neig)

    #find second best match (candidate to be the next best match)
    #coordinate and value
    neig[1,1]=0 #best match temporary forced to 0
    Xmax2, Ymax2=np.unravel_index(neig.argmax(), neig.shape)
    #translate to have relative position to best match
    Xmax2=Xmax2-1
    Ymax2=Ymax2-1
    maxVal2=np.max(neig)
    neig[1,1]=maxVal #revert best match

    #find ponderated barycenter of neighbour matrix
    #the vector Xmoy, Ymoy will be move direction
    Xmoy=(np.sum(neig[:,0])*(-1)+np.sum(neig[:,2])*(1))/np.sum(neig)        
    Ymoy=(np.sum(neig[0,:])*(-1)+np.sum(neig[2,:])*(1))/np.sum(neig)   
    #Xmoy=Xmax2*maxVal2/(maxVal+maxVal2)
    #Ymoy=Ymax2*maxVal2/(maxVal+maxVal2)
    
    if False:
        print(neig)
        print("Max 1",neig[1,1])
        print("Max 2",maxVal)
        print("Xmoy :",Xmoy)
        print("Ymoy :",Ymoy)
        
    #find norm of the displacement
    norm=math.sqrt(Xmoy**2+Ymoy**2)
    
    #method 1 : Linear 
    #ponderated barycenter between best and second best
    #----
    if False:
        newNorm=maxVal/(maxVal+neig[1,1])

    #Method 2 : hyperbolic
    #----
    if True:
        #supposed to be 1 (the best possible match)
        #empirical modif of hyperbol to avoid distorsion
        one=1.5
        K0=maxVal2/(maxVal2+maxVal*((one-maxVal2)/(one-maxVal)))
        #K1=maxVal/(maxVal+maxVal2*((one-maxVal)/(one-maxVal2)))        
        newNorm=K0
    
    dstX=dstX0+Xmoy*newNorm/norm/factor
    dstY=dstY0+Ymoy*newNorm/norm/factor

    #calculate displacement norm
    #---------------------------
    dst=math.sqrt(dstX**2+dstY**2)
    dst0=math.sqrt(dstX0**2+dstY0**2)

    angle=math.atan2(dstY,dstX)*180./math.pi

    quality=maxVal/(maxVal+maxVal2)
    quality=(2-1/quality)

    return (dst,angle)
#DSC06525.JPG	

def ProcessDirectory(path,outputPath,interval=None):


    # fits or DSLR ?
    format = "DSLR"

    for f in listdir(path):
        if isfile(join(path, f)):
            if ".fit" in f or ".FIT" in f:
                format = "FITS"
                print ("FITS file detected")
                break

    print (interval)    

    if interval is None:
        if format == "FITS":
            print ("processing FITS")
            try:
                imagePaths= [[f,GetFitsDate(path+f)] for f in listdir(path) if isfile(join(path, f))]
            except KeyError:
                print("Invalid image file or EXIF data in path")
                raise KeyError
            imagePaths=sorted(imagePaths,key=lambda x: x[1])
    
        else:
            print ("processing EXIF")
            try:
                imagePaths= [[f,GetExifDate(path+f)] for f in listdir(path) if isfile(join(path, f))]
            except KeyError:
                print("Invalid image file or EXIF data in path")
                raise KeyError
            imagePaths=sorted(imagePaths,key=lambda x: x[1])
    else:
            imageFiles= [f for f in listdir(path) if isfile(join(path, f))]
            imageFiles=sorted(imageFiles)
            imagePaths= [[imageFiles[i],i*interval] for i in range(len(imageFiles))]
    results=None


    print ("\nPerforming pattern matching")
    print ("file1\tfile2\tDeltaT\tDist\tAngle")
    for i in range(len(imagePaths)-1):
        date1=imagePaths[i][1]
        date2=imagePaths[i+1][1]
        if interval is None:
            delta=(date2-date1).total_seconds()
        else:
            delta=(date2-date1)
        dist,angle=calcMove(path+imagePaths[i][0],path+imagePaths[i+1][0])
        result=[imagePaths[i][0],
                imagePaths[i+1][0],
                delta,
                dist,angle]
        if results is None:
           results=[result]
        else:
           results.append(result)
        print (imagePaths[i][0]+ "\t"+
               imagePaths[i+1][0]+ "\t"+
               str(delta) + "\t"+
               str(dist)+ "\t"+
               str(angle))

    results=np.array(results)
    resultsDist=np.array([float(dist) for dist in results[:,3]])

    outputRoot=str(datetime.now()).replace(":","-")+"_Eq_calib_"
    outputFile=outputPath+outputRoot+"raw_results.csv"


    print ("saving to "+outputFile)


    np.savetxt(outputFile,
               results,
               delimiter =";",
               fmt ='% s',
               header="file1;file2;DeltaT;dist_Px;angle_Deg",
               comments="")
    return outputFile,outputRoot



if __name__ == '__main__':
    #parser = argparse.ArgumentParser(description='EqDayCalib : day charaterization of Equatorial Mount ')
    #parser.add_argument('-i','--input', help='input directory containing *ONLY* image with valid EXIF', required=True)
    #parser.add_argument('-o','--output', help='output directory to save results file', required=True)
    #args = vars(parser.parse_args())

    #print (args["input"])
    ProcessDirectory(path,"/mnt/data/sandbox/test_image/out")
    # try:
    #     ProcessDirectory(path)
    # except :
    #     print ("Image Analysis failed")