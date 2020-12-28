#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 13:12:05 2020

@author: nicolas
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

path="/mnt/data/sandbox/eq/s1/"

def GetExifDate(imagePath):
    with open(imagePath, 'rb') as fh:
        print(imagePath)
        tags = exifread.process_file(fh, stop_tag="EXIF DateTimeOriginal")
        dateTaken = tags["EXIF DateTimeOriginal"]
        date_obj = datetime.strptime(str(dateTaken), '%Y:%m:%d %H:%M:%S')
        return date_obj


def RemoveOutliers(results,m):
    dist=results[:,1]
    idx = np.where(abs(dist - np.mean(dist)) <= m * np.std(dist))[0]
    results=results[idx]

    angle=results[:,2]
    idx = np.where(abs(angle - np.mean(angle)) <= m * np.std(angle))
    results=results[idx]
    return results


def CalcMove(query,train):
    query_img = cv2.imread(query) 
    train_img = cv2.imread(train) 
    
     #for debug
    width = int(query_img.shape[1] * 0.2)
    height = int(query_img.shape[0] * 0.2)
        # dsize
    dsize = (width, height)     
    #query_img=cv2.resize(query_img,dsize)
    #train_img=cv2.resize(train_img, dsize)
    
    # Convert it to grayscale conda install --name eq
    query_img_bw = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY) 
    train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY) 
    
    #crop query img center
    w = query_img.shape[1] 
    h = query_img.shape[0]
    start=(int(w/2-w*0.1),int(h/2-h*0.1))
    end=(int(w/2+w*0.1),int(h/2+h*0.1)) 
    roi=query_img_bw[start[1]:end[1],start[0]:end[0]]

    #    query_img=cv2.rectangle(query_img,start, end, 255, 2)
    img = train_img_bw
    img2 = img.copy()
    template = roi
    w, h = template.shape[::-1]
    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    
    img = img2.copy()
    method = eval(methods[1])
    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    
    dstX=float(top_left[0]-start[0])
    dstY=float(top_left[1]-start[1])
    
    dst=math.sqrt(dstX**2+dstY**2)
    angle=math.atan2(dstY,dstX)*360/math.pi
    #print (str(dstX)+"\t"+str(dstY)+"\t"+str(dst)+"\t"+str(angle))
    
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
#    if False:
#        cv2.rectangle(img,top_left, bottom_right, 255, 2)
#        plt.subplot(121),plt.imshow(res,cmap = 'gray')
#        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#        plt.subplot(122),plt.imshow(img,cmap = 'gray')
#        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
#        plt.suptitle("")
#       plt.show()
    
    return (dst,0,angle,0)



def ProcessDirectory(path,outputPath):
    #imagePaths=["DSC09572.JPG","DSC09573.JPG"]
    
    print ("processing EXIF")
    try:
        imagePaths= [[f,GetExifDate(path+f)] for f in listdir(path) if isfile(join(path, f))]
    except KeyError:
        print("Invalid image file or EXIF data in path")
        raise KeyError
    imagePaths=sorted(imagePaths,key=lambda x: x[1])
    
    results=None
    
    
    print ("\nPerforming pattern matching")
    print ("file1\tfile2\tDeltaT\tDist\tAngle")
    for i in range(len(imagePaths)-1):
        date1=imagePaths[i][1]
        date2=imagePaths[i+1][1]
        delta=(date2-date1).total_seconds()
        dist,stdDist,angle,stdAngle=CalcMove(path+imagePaths[i][0],path+imagePaths[i+1][0])
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
    
    try:
        ProcessDirectory(path)
    except :
        print ("Image Analysis failed")