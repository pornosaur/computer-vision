#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 12:24:07 2020

@author: patpa
"""

import cv2 as cv
import numpy as np

class OpticalFlow:
    
    def __init__(self):
        pass  
    
    def compute_lucas_kanade(self, img, prev_img):
        g_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        g_prev_img =  cv.cvtColor(prev_img, cv.COLOR_BGR2GRAY)
        
        color = np.random.randint(0,255,(100,3))
        mask = np.zeros_like(g_prev_img)

        f_prev_img = cv.goodFeaturesToTrack(g_prev_img, maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7)
        
        f_img, st, err = cv.calcOpticalFlowPyrLK(g_prev_img, g_img, f_prev_img, None, winSize = (15, 15), maxLevel = 2, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        
        good_new = f_img[st == 1]
        good_old = f_prev_img[st == 1]
    
        for i,(new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv.line(mask, (a,b), (c,d), color[i].tolist(), 2)
            img = cv.circle(img, (a,b), 5, color[i].tolist(), -1)
        final_img = cv.add(img, mask)
        
        cv.imshow('Optical Flow', final_img)
