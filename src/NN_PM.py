from time import time
import numpy as np
import pm3D

from utility import CHF
from utility.utilityImage import img2grayf

def FE_PM(img, param=dict()):
    ## parameters Feature Extraction
    param = setParameters(param,'type_feat',2) # type of feature, one of the following:
        # 1) ZM-cart
        # 2) ZM-polar
        # 3) PCT-cart
        # 4) PCT-polar
        # 5) FMT (log-polar)

    diameter_feat = [12,12,12,12,24]
    param = setParameters(param,'diameter',diameter_feat[param['type_feat']-1]) # patch diameter
    param = setParameters(param,'ZM_order',5)
    param = setParameters(param,'PCT_NM', [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(2,0),(2,1),(3,0)])
    param = setParameters(param,'FMT_N', range(-2,3))
    param = setParameters(param,'FMT_M', range(5))
    param = setParameters(param,'radiusNum', 26) # number of sampling points along the radius
    param = setParameters(param,'anglesNum', 32) # number of sampling points along the circumferences
    param = setParameters(param,'radiusMin',np.sqrt(2.0)) # minimun radius for FMT
    param = setParameters(param,'pad_img', False)


    ## Technique
    img = img2grayf(img)


    print('START\n')
    print('feataure type: %d' % param['type_feat'])
    # (1) Feature Extraction
    timestamp = time()
    # generation of filters
    if param['type_feat']==1:
        bfdata = CHF.ZM_bf(param['diameter'], param['ZM_order'])
    elif param['type_feat']==2:
        bfdata = CHF.ZMp_bf(param['diameter'], param['ZM_order'], param['radiusNum'], param['anglesNum'])
    elif param['type_feat']==3:
        bfdata = CHF.PCT_bf(param['diameter'], param['PCT_NM'])
    elif param['type_feat']==4:
        bfdata = CHF.PCTp_bf(param['diameter'], param['PCT_NM'], param['radiusNum'], param['anglesNum'])
    elif param['type_feat']==5:
        bfdata = CHF.FMTpl_bf(param['diameter'], param['FMT_M'], param['radiusNum'], param['anglesNum'], param['FMT_N'], param['radiusMin'])
    else:
        print('type of feature not found')
        return None

    # border
    raggioU = int(np.ceil((param['diameter']-1.0)/2.0))
    raggioL = int(np.floor((param['diameter']-1.0)/2.0))
    padsize = ((raggioU,raggioL),(raggioU,raggioL))
    if param['pad_img']:
        #print('padding:', img.shape, img.dtype)
        img = np.pad(img.copy(), padsize, mode='edge') #, constant_values = 255.0/256.0)
        #print('padding:', img.shape, img.dtype)
        padsize = ((0,0),(0,0))
        
    # feature generation
    feat = np.abs(CHF.FiltersBank_FFT(img, bfdata, mode='valid'))
    

    timeFE = time() - timestamp
    print('time FE: %0.3f\n' % timeFE)
    return feat, padsize, param['diameter']

def Matching_PM(feat, param=dict()):
    ## Parameters Matching
    param = setParameters(param,'match_num_iter', 8) # N_{it} = number of iterations
    param = setParameters(param,'match_th_dist1', 8) # T_{D1} = minimum length of offsets
    param = setParameters(param,'match_num_tile', 1) # number of thread
    param = setParameters(param,'match_diameter', 1) # block-size

    ## Matching
    timestamp = time()
    feat = (feat-np.min(feat))/(np.max(feat)-np.min(feat)) # mPM requires the features to be in [0,1]
    feat = np.reshape(feat, (feat.shape[0],feat.shape[1],1,feat.shape[2]))
    #feat = feat.astype(np.float32, order='F')
    cnn = pm3D.pm3Dmod(param['match_diameter'], 1, param['match_num_iter'],
                       -param['match_th_dist1'], 0, param['match_num_tile'], feat, feat)
    if param['match_diameter']>1:
        mpfY = cnn[:-(param['match_diameter']-1),:-(param['match_diameter']-1),0,1].astype(np.int16)
        mpfX = cnn[:-(param['match_diameter']-1),:-(param['match_diameter']-1),0,0].astype(np.int16)
    else:
        mpfY = cnn[:,:,0,1].astype(np.int16)
        mpfX = cnn[:,:,0,0].astype(np.int16)
    
    timeMP = time() - timestamp
    print('time PM: %0.3f\n' % timeMP)
    return mpfY, mpfX

def Matching_PM_double(featA, featB, param=dict()):
    ## Parameters Matching
    param = setParameters(param,'match2_num_iter', 16) # N_{it} = number of iterations
    param = setParameters(param,'match2_th_dist1', 0) # T_{D1} = minimum length of offsets
    param = setParameters(param,'match2_num_tile', 1) # number of thread
    param = setParameters(param,'match2_diameter', 1) # block-size
    
    ## Matching
    timestamp = time()
    feat_min = min(np.min(featA), np.min(featB))
    feat_max = max(np.max(featA), np.max(featB))
    featA_rc = (featA-feat_min)/(feat_max-feat_min) # mPM requires the features to be in [0,1]
    featB_rc = (featB-feat_min)/(feat_max-feat_min) # mPM requires the features to be in [0,1]
    featA_rc = np.reshape(featA_rc, (featA_rc.shape[0],featA_rc.shape[1],1,featA_rc.shape[2]))
    featB_rc = np.reshape(featB_rc, (featB_rc.shape[0],featB_rc.shape[1],1,featB_rc.shape[2]))
    
    cnn = pm3D.pm3Dmod(param['match2_diameter'],  1, param['match2_num_iter'], 
                       -param['match2_th_dist1'], 0, param['match2_num_tile'],featA_rc,featB_rc)
    if param['match2_diameter']>1:
        mpfY = cnn[:-(param['match2_diameter']-1),:-(param['match2_diameter']-1),0,1].astype(np.int16)
        mpfX = cnn[:-(param['match2_diameter']-1),:-(param['match2_diameter']-1),0,0].astype(np.int16)
    else:
        mpfY = cnn[:,:,0,1].astype(np.int16)
        mpfX = cnn[:,:,0,0].astype(np.int16)
    
    timeMP = time() - timestamp
    print('time PM: %0.3f\n' % timeMP)
    return mpfY, mpfX

def NN_PM(img, param=dict()):
    feat, padsize, diameter = FE_PM(img, param=param)
    mpfY, mpfX = Matching_PM(feat, param=param)
    
    return mpfY, mpfX, padsize

def setParameters(param,name,value):
    if not(name in param.keys()): param[name] = value
    return param
