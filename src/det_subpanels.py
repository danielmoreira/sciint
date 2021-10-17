import numpy as np
import cv2
import os
from utility.utilityImage import img2grayf
from scipy import ndimage as ndi
from skimage.segmentation import random_walker
from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion, binary_dilation, disk, square



def fill_image(img, poly, color=None):
    img = img.copy()
    flag_compute_color = color is None
    if len(poly)>0:
        poly = np.asarray(list(poly), dtype=np.int32).reshape((len(poly), 1, -1, 1, 2))
        for points in poly:
            msk1 = np.zeros(np.asarray(img).shape[:2], np.uint8)
            msk1 = cv2.fillPoly(msk1, points, 255)>128
            if flag_compute_color:
                msk2 = np.zeros(np.asarray(img).shape[:2], np.uint8)
                msk2 = cv2.polylines(msk2, points, True, color=255, thickness=2)>128
                
                if len(img.shape)==2:
                    color = np.median(img[msk2])
                else:
                    color = [np.median(img[msk2,index]) for index in range(img.shape[-1])]
            
            if len(img.shape)==2:
                 img[msk1] = color
            else:
                for index in range(img.shape[-1]):
                    img[msk1, index] = color[index]

    return img

def get_mask(img, poly):
    return fill_image(np.zeros(np.asarray(img).shape[:2], np.uint8), poly, 255)

def get_mask_panels(img, panels):
    y = np.zeros(np.asarray(img).shape[:2], np.uint8)
    for panel in panels:
        y[int(panel[1]):int(panel[3]), int(panel[0]):int(panel[2])] = 255 
    return y

def hysteresis(mask_high, mask_low, s=8):
    if s is None:
        s = disk(1)
    elif isinstance(s, int) or s.size == 1:
        if s == 8:
            s = square(3)
        else:
            s = disk(s // 4)

    mask = np.logical_and(binary_dilation(mask_high, s), mask_low)
    while not np.equal(mask, mask_high).all():
        mask_high = mask
        mask = np.logical_and(binary_dilation(mask_high, s), mask_low)
    return mask
    
def subpanel_segmentation(img_np, poly, background_th=0.98, foreground_th_low=0.95, foreground_th_high=0.7, min_cc_size=32*32):
    # Convert image to grayscale
    if len(img_np.shape) > 2:
        img_gr = img2grayf(img_np)
    else:
        img_gr = img_np.copy()
    
    if poly is not None:
        img_gr = fill_image(img_gr.copy(), poly, color=1.0)
    
    # Perform random walker segmentation
    msk_background = img_gr > background_th
    msk_foreground = hysteresis(img_gr < foreground_th_high, img_gr < foreground_th_low)
    l_min = int(np.sqrt(min_cc_size)/4)
    msk_foreground = binary_erosion(msk_foreground, square(l_min))

    ind_background, ind_foreground = 1, 2
    markers = np.zeros(img_gr.shape, dtype=np.uint)
    markers[msk_background] = ind_background
    markers[msk_foreground] = ind_foreground
    labels = random_walker(img_gr, markers, beta=50, mode='bf')
    
    msk = labels == ind_foreground
    l_min = int(np.sqrt(min_cc_size)/2)
    msk = hysteresis(binary_erosion(msk, square(l_min)), msk, 4)
    
    lab = label(msk)
    panels_list = list()
    for p in regionprops(lab):
        region = msk[p.bbox[0]:p.bbox[2], p.bbox[1]:p.bbox[3]]
        
        if max((region.mean(0) > .5).mean(), (region.mean(1) > .5).mean()) >= .6:
            if (np.sum(region)>=min_cc_size): 
                panels_list.append((p.bbox[1], p.bbox[0], p.bbox[3], p.bbox[2]))
    
    return panels_list
