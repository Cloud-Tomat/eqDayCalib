#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 20:55:04 2021

@author: nicolas
"""

import numpy as np
import cv2 



import matplotlib.pyplot as plt

class cFrame:
    def __init__(self,timeStamp,img,shift):
        self.frame=img
        self.timeStamp=timeStamp
        self.shift=None
    
class cFrameBuffer:
    #Number of frame to compare (key frames)
    bufferSize=1
    #distance between key frames
    distLimit=5
    
    #displacement direction 
    sizeX=400
    sizeY=400
    
    
    debug=False
    
    def __init__(self,timeStamp,img):        
        #-------------------
        self.refFrame=None
        self.shiftAccu=np.array([0.,0.])
        self.refFrame=cFrame(timeStamp,img,np.array([0.,0.]))
    
    def crop(self,frame,sizeX,sizeY,offsetX=0,offsetY=0):
        w = frame.shape[1] 
        h = frame.shape[0]
        start=(int(w/2-sizeX/2)-offsetX,int(h/2-sizeY/2)-offsetY)
        end=(int(w/2+sizeX/2)-offsetX,int(h/2+sizeY/2)-offsetY) 
        frameCrop=frame[start[1]:end[1],start[0]:end[0]]
        return frameCrop
   
    def update(self,timeStamp,img,alt=False):

        shift=self.calcMove(self.refFrame.frame,img)
 
        norm=np.linalg.norm(shift)
        normAccu=np.linalg.norm(shift+self.shiftAccu)
        
        if norm>(self.distLimit):
            self.refFrame=cFrame(timeStamp,img,np.array([0.,0.]))
            self.shiftAccu=self.shiftAccu+shift
            print("----SWAP----------")

        print(shift)

        return (normAccu,0,0,0) 

   
    
    def calcMove(self,im1,im2):
        if self.debug:
           fig = plt.figure(figsize=(8, 3))
           ax1 = plt.subplot(1, 3, 1)
           ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
           ax3 = plt.subplot(1, 3, 3)
        
        #im1=self.crop(im1,self.sizeX,self.sizeY)
        #im2=self.crop(im2,self.sizeX,self.sizeY)
        
        im1=im1.astype(np.float32)
        im2=im2.astype(np.float32)
        
        
        mapper = cv2.reg_MapperGradShift()
        mappPyr = cv2.reg_MapperPyramid(mapper)
        
        resMap = mappPyr.calculate(im1, im2)
        mapShift = cv2.reg.MapTypeCaster_toShift(resMap)
        solution=mapShift.getShift()
        shift=[solution[0][0],solution[1][0]]
        
        if self.debug:        
            ax1.imshow(im1, cmap='gray')
            ax1.set_axis_off()
            ax1.set_title('Reference image')
            
            ax2.imshow(im2.real, cmap='gray')
            ax2.set_axis_off()
            ax2.set_title('Offset image')
            plt.show()        
        
        
        return np.array(shift)
 
    
    
