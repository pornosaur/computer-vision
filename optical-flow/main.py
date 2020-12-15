#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 12:22:51 2020

@author: patpa
"""

import opticalflow as of
import cv2 as cv


def main():
    flow = of.OpticalFlow()
    
    img1 = cv.imread("data/frame10.png", cv.IMREAD_COLOR)
    img2 = cv.imread("data/frame11.png", cv.IMREAD_COLOR)
    
    flow.compute_horn_schunck(img1, img2)
    

if __name__ == "__main__":
    main()
    