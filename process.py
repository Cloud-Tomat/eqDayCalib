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


#read and preprocess CSV
df=pd.read_csv(path+"result_step6.csv",sep=";") 
timeStep=np.mean(df['DeltaT'])
speeds=df["dist_Px"]/timeStep
scale=SIDERAL_SPEED/np.mean(speeds)
speeds=speeds*scale
print(np.mean(speeds))


print(scale)
print(timeStep)

integrale=0.0
positions=np.array([])
for i in range(len(speeds)):
    integrale+=float(speeds[i])*timeStep
    positions=np.hstack((positions,integrale))

#print (posPx)

x=np.arange(0,len(positions))*timeStep
#print(x)

z = np.polyfit(x,positions , 2)
p = np.poly1d(z)
#print(z)

#Periodic error
eps=np.array([positions[i]-p(x[i]) for i in range(len(positions))])


#Compute Discrete Fourrier Transform
epFfts = np.absolute(np.fft.fft(eps))
freqs=np.arange(int(len(epFfts)/2))/(len(epFfts)*x[-1]/(len(x)-1))
periods=np.array([1/f for f in freqs[:-1])
epFfts=epFfts[:-1]
#np.transpose([epPx,epFft])

#print([epPx[:5],epFft[:5]])
#print(epPx)

epData=pd.DataFrame({"time":x,"EP":eps})
fig = px.line(epData,x="time",y="EP")
fig.show()

dftData=pd.DataFrame({"period(s)":periods,"EP_FFT":epFfts[:len(periods)]})
fftFig = px.line(dftData,x="period(s)",y="EP_FFT")
fftFig.show()


if False:
    np.savetxt(path+"temp.csv",  
               epFfts[:int(len(epFfts)/2)], 
               delimiter =";",  
               fmt ='% s',
               comments="") 