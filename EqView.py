#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 23:56:16 2020

@author: nicolas
"""

import track_feature_match
import process
import argparse
import pandas as pd

import plotly.express as px
import plotly.io as pio
import pandas as pd

pio.renderers.default='browser'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EqDayCalib Viewer')
    parser.add_argument('-i','--input', help='input file to view', required=True)
    args = vars(parser.parse_args())

    try:
        inputPath=args["input"]
    except:
        print("Invalid Arguments")
        exit()

    #try:
    df=pd.read_csv(inputPath,sep=";") 
    print(df.columns[1])
    fig = px.line(df,x=df.columns[1],y=df.columns[2])
    fig.show()
    #except :
    #    print ("Image Analysis failed")
    #    exit()
        
    