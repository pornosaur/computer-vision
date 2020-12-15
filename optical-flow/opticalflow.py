#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 12:24:07 2020

@author: patpa

Horn-Schunck impl. - https://github.com/scivision/
Others - https://nanonets.com/blog/optical-flow/
"""

import cv2 as cv
import numpy as np
from scipy.ndimage.filters import convolve as filter2
from matplotlib import pyplot as plt

kernelX = np.array([[-1, 1], [-1, 1]]) * .25  # kernel for computing d/dx

kernelY = np.array([[-1, -1],
                    [1, 1]]) * .25  # kernel for computing d/dy

kernelT = np.ones((2, 2))*.25

class OpticalFlow:
    
    def __init__(self):
        pass  
    
    def flownet2(self, img1, img2):
        pass
    
    def compute_horn_schunck(self, im1, im2, alpha = 1.0, Niter = 8):
        g_im1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
        g_im2 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
       
        hsv = np.zeros_like(im1)
        # Sets image saturation to maximum
        hsv[..., 1] = 255

        # set up initial velocities
        uInitial = np.zeros([g_im1.shape[0], g_im1.shape[1]])
        vInitial = np.zeros([g_im1.shape[0], g_im1.shape[1]])
            
        # Set initial value for the flow vectors
        U = uInitial
        V = vInitial
    
        # Estimate derivatives
        [fx, fy, ft] = self.compute_derivatives(g_im1, g_im2)
        
        kernel = np.matrix([[1/12, 1/6, 1/12],[1/6, 0, 1/6],[1/12, 1/6, 1/12]])

        # Iteration to reduce error
        for _ in range(Niter):
            # Compute local averages of the flow vectors
            uAvg = filter2(U, kernel)
            vAvg = filter2(V, kernel)
            
            der = (fx*uAvg + fy*vAvg + ft) / (alpha**2 + fx**2 + fy**2)
            U = uAvg - fx * der
            V = vAvg - fy * der

        mag, ang = cv.cartToPolar(U, V)
        hsv[...,0] = ang * 180/np.pi/2
        hsv[...,2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        
      #  cv.imshow("Horn-Schunck Optical Flow", rgb)
       # cv.waitKey()
        #cv.destroyAllWindows()
    

    def compute_derivatives(self, im1, im2):
        fx = filter2(im1, kernelX) + filter2(im2, kernelX)
        fy = filter2(im1, kernelY) + filter2(im2, kernelY)
        ft = filter2(im1, kernelT) + filter2(im2, -kernelT)
    
        return fx, fy, ft
    
    def compute_farnback(self, img, prev_img):
        hsv = np.zeros_like(prev_img)
        hsv[...,1] = 255
        
        g_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        g_prev_img = cv.cvtColor(prev_img, cv.COLOR_BGR2GRAY)
        f_img = cv.calcOpticalFlowFarneback(g_prev_img, g_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        mag, ang = cv.cartToPolar(f_img[...,0], f_img[...,1])
        hsv[...,0] = ang * 180/np.pi/2
        hsv[...,2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        
        cv.imshow("Farnback Optical Flow", rgb)
        cv.waitKey()
        cv.destroyAllWindows()
        
        return rgb.copy()
        
    
    def compute_lucas_kanade(self, img, prev_img):
        g_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        g_prev_img =  cv.cvtColor(prev_img, cv.COLOR_BGR2GRAY)
        
        color = np.random.randint(0,255,(100,3))
        mask = np.zeros_like(prev_img)

        f_prev_img = cv.goodFeaturesToTrack(g_prev_img, maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7)
        
        f_img, st, err = cv.calcOpticalFlowPyrLK(g_prev_img, g_img, f_prev_img, None, winSize = (15, 15), maxLevel = 2, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        
        good_new = f_img[st == 1]
        good_old = f_prev_img[st == 1]
    
        for i,(new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv.line(mask, (a,b), (c,d), color[i].tolist(), 2)
           # img = cv.circle(img, (a,b), 5, color[i].tolist(), -1)
        final_img = cv.add(img, mask)
        
        cv.imshow('Optical Flow', final_img)
        cv.waitKey()
        
        cv.destroyAllWindows()
        
        return final_img.copy()
