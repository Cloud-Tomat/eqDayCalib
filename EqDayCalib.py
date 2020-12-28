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
    args = vars(parser.parse_args())

    try:
        inputPath=args["input"]+"/"
        outputPath=args["output"]+"/"
    except:
        print("Invalid Arguments")
        exit()

    #try:
    outputFile,outputRoot=track_feature_match.ProcessDirectory(inputPath,outputPath)
    process.Process(outputFile,outputPath,outputRoot)
    #except :
    #    print ("Image Analysis failed")
    #    exit()
        
    