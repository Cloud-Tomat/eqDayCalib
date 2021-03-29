#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 22:38:58 2020

@author: nicolas
"""

import numpy as np
import csv


import plotly.express as px
import plotly.io as pio
import pandas as pd

pio.renderers.default='browser'


path="/mnt/data/sandbox/eq/"
SIDERAL_SPEED=15.035

def Process (path,outputPath,rootPath):
    print("\nPost-processing Data")

    #read and preprocess CSV
    df=pd.read_csv(path,sep=";") 
    # print(df)
    # df=df[:60]
    
    timeStep=np.mean(df['DeltaT'])
    speeds=df["dist_Px"]/timeStep
    
    #Create x (time axis)
    x=np.arange(0,len(speeds))*timeStep
    
    #Evaluate Constants
    scale=SIDERAL_SPEED/np.mean(speeds)
    speeds=speeds*scale

        
    print("Scale (Arcec/pixel) : " + str(scale))
    print("Shoot interval(s) : "+ str(timeStep))
    
    #Perform Speed integration to retrieve position
    integrale=0.0
    positions=np.array([])
    for i in range(len(speeds)):
        integrale+=float(speeds[i])*timeStep
        positions=np.hstack((positions,integrale))
    
    #Polynominal regression to remove contant and compensate integraiton drif
    z = np.polyfit(x,positions , 2)
    p = np.poly1d(z)
    
    #Periodic error (Postion - poly regression)
    eps=np.array([positions[i]-p(x[i]) for i in range(len(positions))])
    
    
    #Compute Discrete Fourrier Transform
    epFfts = np.absolute(np.fft.fft(eps))
    freqs=np.arange(int(len(epFfts)/2))/(len(epFfts)*x[-1]/(len(x)-1))
    periods=np.array([1/f for f in freqs[1:]])
    epFfts=epFfts[1:]/(len(epFfts)/2)
    
    #Grasph
    epData=pd.DataFrame({"time(s)":x,"EP(ArcSec)":eps})
    fig = px.line(epData,x="time(s)",y="EP(ArcSec)")
    fig.show()
    
    dftData=pd.DataFrame({"period(s)":periods,"EP_FFT(ArcSec)":epFfts[:len(periods)]})
    fftFig = px.line(dftData,x="period(s)",y="EP_FFT(ArcSec")
    fftFig.show()
    
    outputFile=outputPath+"/"+rootPath
    print(outputFile)
    
    epData.to_csv(outputFile+"scaled_EP.csv",sep=";")
    dftData.to_csv(outputFile+"EP_DFT.csv",sep=";")


if __name__ == '__main__':
    outputFile="/mnt/data/sandbox/test_image/out/2021-03-29 21-37-42.722138_Eq_calib_raw_results.csv"
    outputRoot="2021-03-27 22-04-20.155802_Eq_calib_"
    outputPath="/mnt/data/sandbox/test_image/out/"
    Process(outputFile,outputPath,outputRoot)