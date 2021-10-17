import os
import sys
from PIL import Image
import numpy as np
sys.path.append(os.path.dirname(__file__))

from utility.utilityImage import imread2f, img2grayf
from NN_PM import setParameters
from core import alg_single, alg_double, get_colored, alg_match_single, alg_match_double
from det_subpanels import get_mask, subpanel_segmentation
from text_detector import TextNet

def ext_pannels_padding(panels, padsize): 
    return [ (panel[0]+padsize[1][0], panel[1]+padsize[0][0], panel[2]-padsize[1][1], panel[3]-padsize[0][1]) for panel in panels]


def alg_single_rgb(filename_img, param = dict()):
    param['remove_panel'] = False
    param = setParameters(param, 'subpanel_background_th'     , 0.98)
    param = setParameters(param, 'subpanel_foreground_th_low' , 0.95)
    param = setParameters(param, 'subpanel_foreground_th_high', 0.70)
    param = setParameters(param, 'subpanel_min_cc_size'       , 32*32)
    param = setParameters(param, 'match_diameter' , 9)
    param['match2_diameter'] = param['match_diameter']
    
    input_img = imread2f(filename_img, channel=3)
    img = input_img.copy()
    texts = TextNet(cuda=False)(np.uint8(256*img))[0]
    panels = subpanel_segmentation(img, texts, background_th=param['subpanel_background_th'],
                                   foreground_th_low =param['subpanel_foreground_th_low'],
                                   foreground_th_high=param['subpanel_foreground_th_high'],
                                   min_cc_size=param['subpanel_min_cc_size'])
    if len(panels)==0:
        panels = subpanel_segmentation(img, texts, foreground_th_low=255.0/256,  background_th=255.0/256,
                                    foreground_th_high=param['subpanel_foreground_th_high'],
                                    min_cc_size=param['subpanel_min_cc_size'])
    
    num_panels = len(panels)
    msk_text = get_mask(img, texts)==0
    
    # Feature Extraction & Matching
    feat = input_img.copy()
    raggioU = int(np.ceil((param['match_diameter']-1.0)/2.0))
    raggioL = int(np.floor((param['match_diameter']-1.0)/2.0))
    padsize = ((raggioU,raggioL),(raggioU,raggioL))
    
    feats     = [feat[int(panel[1]):int(panel[3]), int(panel[0]):int(panel[2]), ...] for panel in panels]
    panels = ext_pannels_padding(panels, padsize)
    msks_text = [msk_text[int(panel[1]):int(panel[3]), int(panel[0]):int(panel[2])] for panel in panels]
    
    
    if len(panels)>0:
        th_size = min(max(int(np.max([x.shape[0]*x.shape[1] for x in msks_text])/100),100),300)
        param = setParameters(param, 'th_sizeA', th_size)
        print('th_size:', param['th_sizeA'])
    
    msk_tot = np.zeros(msk_text.shape[:2], np.bool)
    for index1 in range(num_panels):
        for index2 in range(num_panels):
            print(index1,index2,num_panels)
            if index2==index1:
                msk1 = alg_match_single(feats[index1], input_mask = msks_text[index1], param = param)
            else:
                msk1, msk2 = alg_match_double(feats[index1], feats[index2],
                     input_maskA = msks_text[index1],
                     input_maskB = msks_text[index2],
                     param = param)
                
                panel = panels[index2]
                msk_tot[int(panel[1]):int(panel[3]), int(panel[0]):int(panel[2])] |= msk2

            panel = panels[index1]
            msk_tot[int(panel[1]):int(panel[3]), int(panel[0]):int(panel[2])] |= msk1

    #msk_tot = np.pad(msk_tot, padsize, 'constant', constant_values=False)
    
    return msk_tot


def alg_double_rgb(filename_imgA, filename_imgB, param = dict()):
    param['remove_panel'] = False
    param = setParameters(param, 'subpanel_background_th'     , 0.98)
    param = setParameters(param, 'subpanel_foreground_th_low' , 0.95)
    param = setParameters(param, 'subpanel_foreground_th_high', 0.70)
    param = setParameters(param, 'subpanel_min_cc_size'       , 32*32)
    param = setParameters(param, 'match_diameter' , 9)
    param['match2_diameter'] = param['match_diameter']
    
    text_det = TextNet(cuda=False)
    input_imgA, input_imgB = imread2f(filename_imgA, channel=3), imread2f(filename_imgB, channel=3)
    textsA = text_det(np.uint8(256*input_imgA))[0]
    textsB = text_det(np.uint8(256*input_imgB))[0]
    panelsA = subpanel_segmentation(input_imgA, textsA, background_th=param['subpanel_background_th'],
                                   foreground_th_low =param['subpanel_foreground_th_low'],
                                   foreground_th_high=param['subpanel_foreground_th_high'],
                                   min_cc_size=param['subpanel_min_cc_size'])
    panelsB = subpanel_segmentation(input_imgB, textsB, background_th=param['subpanel_background_th'],
                                   foreground_th_low =param['subpanel_foreground_th_low'],
                                   foreground_th_high=param['subpanel_foreground_th_high'],
                                   min_cc_size=param['subpanel_min_cc_size'])
    if len(panelsA)==0:
        panelsA = subpanel_segmentation(input_imgA, textsA, foreground_th_low=255.0/256,  background_th=255.0/256,
                                    foreground_th_high=param['subpanel_foreground_th_high'],
                                    min_cc_size=param['subpanel_min_cc_size'])
    if len(panelsB)==0:
        panelsB = subpanel_segmentation(input_imgB, textsB, foreground_th_low=255.0/256,  background_th=255.0/256,
                                    foreground_th_high=param['subpanel_foreground_th_high'],
                                    min_cc_size=param['subpanel_min_cc_size'])
    
    
    msk_textA = get_mask(input_imgA, textsA)==0
    msk_textB = get_mask(input_imgB, textsB)==0
    
    # Feature Extraction & Matching
    featA = input_imgA.copy()
    featB = input_imgB.copy()
    raggioU = int(np.ceil((param['match_diameter']-1.0)/2.0))
    raggioL = int(np.floor((param['match_diameter']-1.0)/2.0))
    padsize = ((raggioU,raggioL),(raggioU,raggioL))
    
    featsA = [featA[int(panel[1]):int(panel[3]), int(panel[0]):int(panel[2]), ...] for panel in panelsA]
    featsB = [featB[int(panel[1]):int(panel[3]), int(panel[0]):int(panel[2]), ...] for panel in panelsB]
    panelsA = ext_pannels_padding(panelsA, padsize)
    panelsB = ext_pannels_padding(panelsB, padsize)
    msks_textA = [msk_textA[int(panel[1]):int(panel[3]), int(panel[0]):int(panel[2])] for panel in panelsA]
    msks_textB = [msk_textB[int(panel[1]):int(panel[3]), int(panel[0]):int(panel[2])] for panel in panelsB]
    
    areas = [x.shape[0]*x.shape[1] for x in msks_textA] + [x.shape[0]*x.shape[1] for x in msks_textB] 
    if len(areas)>0:
        th_size = min(max(int(np.max(areas)/100),100),300)
        param = setParameters(param, 'th_sizeA', th_size)
        print('th_size:', param['th_sizeA'])
    
    msk_totA = np.zeros(msk_textA.shape[:2], np.bool)
    msk_totB = np.zeros(msk_textB.shape[:2], np.bool)
    for indexA in range(len(featsA)):
        for indexB in range(len(featsB)):
            print(indexA, indexB, len(featsA), len(featsB))
            mskA, mskB = alg_match_double(featsA[indexA], featsB[indexB],
                     input_maskA = msks_textA[indexA], input_maskB = msks_textB[indexB],
                     param=param)
                
            panel = panelsA[indexA]
            msk_totA[int(panel[1]):int(panel[3]), int(panel[0]):int(panel[2])] |= mskA
            panel = panelsB[indexB]
            msk_totB[int(panel[1]):int(panel[3]), int(panel[0]):int(panel[2])] |= mskB

    #msk_totA = np.pad(msk_totA, padsizeA, 'constant', constant_values=False)
    #msk_totB = np.pad(msk_totB, padsizeB, 'constant', constant_values=False)
    
    return msk_totA, msk_totB

    
if __name__=="__main__":
    from time import time
    if len(sys.argv)==3:
        filename_img = sys.argv[1]  # input image file name
        filename_out = sys.argv[2]  # output image file name
        filename_out, filename_ext = os.path.splitext(filename_out)
        if filename_ext=='.npz': filename_ext = '.png'
        filename_out_dat = filename_out + '.npz'
        filename_out_msk = filename_out + filename_ext
        filename_out_col = filename_out + '_col' + filename_ext
        
        timestamp = time()
        img = imread2f(filename_img, channel=3)
        mask = alg_single_rgb(filename_img)
        colored_output = get_colored(img, mask)
        if np.any(mask==0):
            score = 1.0
        else:
            score = 0.0
        timeApproach = time() - timestamp

        img_size = Image.open(filename_img).size
        outmat = dict()
        outmat['map'] = mask
        outmat['score'] = score
        outmat['imgsize'] = (img_size[1], img_size[0])
        outmat['time'] = timeApproach
        try:
            os.makedirs(os.path.dirname(filename_out_dat), exist_ok=True)
        except:
            pass
        np.savez(filename_out_dat, **outmat)
        Image.fromarray(colored_output).save(filename_out_col)
        mask = 255 * np.logical_not(mask).astype(np.uint8)
        Image.fromarray(mask).save(filename_out_msk)
        
    elif len(sys.argv)==4:
        filename_imgA = sys.argv[1]  # input image file name
        filename_imgB = sys.argv[2]  # input image file name
        filename_out  = sys.argv[3]  # output image file name
        filename_out, filename_ext = os.path.splitext(filename_out)
        if filename_ext=='.npz': filename_ext = '.png'
        filename_out_dat  = filename_out + '.npz'
        filename_out_mskA = filename_out + '_A' + filename_ext
        filename_out_mskB = filename_out + '_B' + filename_ext
        filename_out_colA = filename_out + '_A_col' + filename_ext
        filename_out_colB = filename_out + '_B_col' + filename_ext
        
        timestamp = time()
        imgA, imgB = imread2f(filename_imgA, channel=3), imread2f(filename_imgB, channel=3)
        maskA, maskB = alg_double_rgb(filename_imgA, filename_imgB)
        colA, colB = get_colored(imgA, maskA),  get_colored(imgB, maskB)
        if np.any(maskA==0):
            score = 1.0
        else:
            score = 0.0
        timeApproach = time() - timestamp

        outmat = dict()
        outmat['mapA'] = maskA
        outmat['mapB'] = maskB
        outmat['score'] = score
        outmat['time'] = timeApproach
        try:
            os.makedirs(os.path.dirname(filename_out_dat), exist_ok=True)
        except:
            pass
        np.savez(filename_out_dat, **outmat)
        Image.fromarray(colA).save(filename_out_colA)
        Image.fromarray(colB).save(filename_out_colB)
        maskA = 255 * np.logical_not(maskA).astype(np.uint8)
        maskB = 255 * np.logical_not(maskB).astype(np.uint8)
        Image.fromarray(maskA).save(filename_out_mskA)
        Image.fromarray(maskB).save(filename_out_mskB)
        
    else:
        print('number of parameters is not correct')