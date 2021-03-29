#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 23:56:16 2020

@author: nicolas
"""

import track_feature_match
import process
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EqDayCalib : day charaterization of Equatorial Mount ')
    parser.add_argument('-i','--input', help='input directory containing *ONLY* image with valid EXIF', required=True)
    parser.add_argument('-o','--output', help='output directory to save results file', required=True)
    parser.add_argument('-t','--time',help='Optional time step between shots, exif data ignored is specified',required=False)
    args = vars(parser.parse_args())

    try:
        inputPath=args["input"]+"/"
        outputPath=args["output"]+"/"
        if args["time"] is not None:
            timeStep=float(args["time"])
        else:
            timeStep=None
    except:
        print("Invalid Arguments")
        exit()

    #try:
        
    outputFile,outputRoot=track_feature_match.ProcessDirectory(inputPath,outputPath,interval=timeStep)
    process.Process(outputFile,outputPath,outputRoot)
    print(outputFile,outputRoot)
    #except :
    #    print ("Image Analysis failed")
    #    exit()
        
    