import os
from utility.utilityImage import imread2f, img2grayf
import utility.CMFD_PM_utily as utl
from NN_PM import FE_PM, Matching_PM, Matching_PM_double, setParameters
from PIL import Image
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes

def clip_padding(x, padsize):
    return x[padsize[0][0]:(x.shape[0]-padsize[0][1]), padsize[1][0]:(x.shape[1]-padsize[1][1])]

def clip_pannels_padding(panels, padsize): 
    return [ (max(panel[0]-padsize[1][0],0), max(panel[1]-padsize[0][0],0), panel[2], panel[3]) for panel in panels]
    
def local_variance(img, diameter):
    import scipy.ndimage as simg
    m0 = simg.uniform_filter(img    , (diameter,diameter), mode = 'constant', cval = 1.0)
    m0 = simg.uniform_filter(img*img, (diameter,diameter), mode = 'constant', cval = 1.0) - m0*m0    
    return m0 

def detect_out_of_panel(img):
    # Assuming that the figure background is white, locate rectangular foreground objects
    from skimage.measure import label, regionprops
    from skimage.morphology import binary_erosion, square
    from scipy.signal import medfilt2d
    def keep_compact(mask_low, s=8):
        mask_high = binary_erosion(mask_low, square(19))
        from skimage.morphology import binary_dilation, disk
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

    if img.max() <= 1.:
        foreground = img < 250./255.
    else:
        foreground = img < 250
    if foreground.ndim == 3:
        foreground = foreground.any(2)
    foreground = binary_erosion(foreground, square(3))

    lab = label(keep_compact(foreground))
    out = np.zeros_like(lab, np.bool)
    for p in regionprops(lab):
        bbox = foreground[p.bbox[0]:p.bbox[2], p.bbox[1]:p.bbox[3]]
        if max((bbox.mean(0) > .75).mean(), (bbox.mean(1) > .75).mean()) >= .8:
            out[p.bbox[0]:p.bbox[2], p.bbox[1]:p.bbox[3]] = True

    return out

def remove_out_of_panel(mask_in, img):
    out = detect_out_of_panel(img)
    mask_out = np.logical_and(mask_in, out)
    return mask_out


def alg_single(input_img, input_mask=None, param = dict()):
    param = setParameters(param, 'remove_panel', True)

    img = img2grayf(input_img.copy())
    # Feature Extraction & Matching
    feat, padsize, diameter = FE_PM(img, param=param)
    if param['remove_panel']:
        print('remove single')
        # removal of regions outside the panels
        remove_msk = clip_padding(detect_out_of_panel(img), padsize)
    else:
        remove_msk = None
    
    if input_mask is not None:
        input_mask = clip_padding(input_mask, padsize)
    
    mask = alg_match_single(feat, input_mask = input_mask, remove_msk = remove_msk, param = param)
    mask = np.pad(mask, padsize, 'constant', constant_values=False)
    return mask

def alg_match_single(feat, input_mask = None, remove_msk = None, param = dict()):
    param = setParameters(param, 'th2_dist2', 10 * 10)  # T^2_{D2} = minimum diatance between clones
    param = setParameters(param, 'th2_dlf', 300)  # T^2_{\epsion} = threshold on DLF error
    param = setParameters(param, 'th_scale', 0.05)  # 
    param = setParameters(param, 'th_sizeA', 300)  # T_{S} = minimum size of clones
    param = setParameters(param, 'th_sizeB', param['th_sizeA'])  # T_{S} = minimum size of clones
    param = setParameters(param, 'rd_median', 4)  # \rho_M = radius of median filter
    param = setParameters(param, 'rd_dlf', 6)  # \rho_N = radius of DLF patch
    param = setParameters(param, 'rd_dil', param['rd_dlf'] + param['rd_median'])  # \rho_D = radius for dilatetion
    
    mpfY, mpfX = Matching_PM(feat.copy(), param=param)
    
    mpfY = mpfY.astype(np.float64)
    mpfX = mpfX.astype(np.float64)
    
    # regularize offsets field by median filtering
    mpfY, mpfX = utl.MPFregularize(mpfY, mpfX, param['rd_median'])
    
    # Compute the squared error of dense linear fitting
    DLFerr = utl.MPF_DLFerror(mpfY, mpfX, param['rd_dlf'])    
    DLFscale = utl.MPF_DLFscale(mpfY, mpfX, param['rd_dlf'])
    mask   = (DLFerr <= param['th2_dlf']) & (DLFscale > param['th_scale'])
    if input_mask is not None:
        mask = mask & input_mask
    
    # removal of close couples
    dist2 = utl.MPFspacedist2(mpfY, mpfX)
    mask  = mask & (dist2 >= param['th2_dist2'])
    
    # removal of small regions
    mask = utl.removesmall(mask, param['th_sizeA'], connectivity=8)
    mask = utl.MPFdual(mpfY, mpfX, mask)  # mirroring of detected regions
    mask = utl.removesmall(mask, param['th_sizeB'], connectivity=8)
    
    if remove_msk is not None:
        # removal of regions outside the panels
        mask = mask & remove_msk
    
    # removal regions without mirror
    msk = utl.dilateDisk(mask, param['rd_median'])
    msk, mask = utl.MPFmirror(mpfY.astype(np.int32), mpfX.astype(np.int32), mask, msk)
    mask = (mask==1) | (msk==1)
    
    mask = utl.dilateDisk(mask, param['rd_dil'])
    mask = binary_fill_holes(mask)
    
    
    return mask

def alg_double(input_imgA, input_imgB, input_maskA=None, input_maskB=None, param = dict()):
    param = setParameters(param, 'remove_panel', True)
    
    imgA = img2grayf(input_imgA.copy())
    imgB = img2grayf(input_imgB.copy())
    
    # Feature Extraction & Matching
    featA, padsizeA, diameterA = FE_PM(imgA, param=param)
    featB, padsizeB, diameterB = FE_PM(imgB, param=param)
    if param['remove_panel']:
        # removal of regions outside the panels
        print('remove double')
        remove_mskA = clip_padding(detect_out_of_panel(imgA), padsizeA)
        remove_mskB = clip_padding(detect_out_of_panel(imgB), padsizeB)
    else:
        remove_mskA = None
        remove_mskB = None
        
    if input_maskA is not None:
        input_maskA = clip_padding(input_maskA, padsizeA)
    if input_maskB is not None:
        input_maskB = clip_padding(input_maskB, padsizeB)
    
    maskA, maskB = alg_match_double(featA, featB,
                     input_maskA = input_maskA, input_maskB = input_maskB,
                     remove_mskA = remove_mskA, remove_mskB = remove_mskB, param = param)
    
    maskA = np.pad(maskA, padsizeA, 'constant', constant_values=False)
    maskB = np.pad(maskB, padsizeB, 'constant', constant_values=False)
    
    return maskA, maskB

def alg_match_double(featA, featB,
                     input_maskA = None, input_maskB = None,
                     remove_mskA = None, remove_mskB = None, param = dict()):
    
    param = setParameters(param, 'th2_dist2', 10 * 10)  # T^2_{D2} = minimum diatance between clones
    param = setParameters(param, 'th2_dlf', 300)  # T^2_{\epsion} = threshold on DLF error
    param = setParameters(param, 'th_scale', 0.05)  # 
    param = setParameters(param, 'th_sizeA', 300)  # T_{S} = minimum size of clones
    param = setParameters(param, 'th_sizeB', param['th_sizeA'])  # T_{S} = minimum size of clones
    param = setParameters(param, 'rd_median', 4)  # \rho_M = radius of median filter
    param = setParameters(param, 'rd_dlf', 6)  # \rho_N = radius of DLF patch
    param = setParameters(param, 'rd_dil', param['rd_dlf'] + param['rd_median'])  # \rho_D = radius for dilatetion

    mpfYA, mpfXA = Matching_PM_double(featA.copy(), featB.copy(), param=param)
    mpfYB, mpfXB = Matching_PM_double(featB.copy(), featA.copy(), param=param)
    
    mpfYA = mpfYA.astype(np.float64); mpfXA = mpfXA.astype(np.float64)
    mpfYB = mpfYB.astype(np.float64); mpfXB = mpfXB.astype(np.float64)
    
    # regularize offsets field by median filtering
    mpfYA, mpfXA = utl.MPFregularize(mpfYA, mpfXA, param['rd_median'])
    mpfYB, mpfXB = utl.MPFregularize(mpfYB, mpfXB, param['rd_median'])
    
    # Compute the squared error of dense linear fitting
    DLFerrA   = utl.MPF_DLFerror(mpfYA, mpfXA, param['rd_dlf'])
    DLFscaleA = utl.MPF_DLFscale(mpfYA, mpfXA, param['rd_dlf'])
    DLFerrB   = utl.MPF_DLFerror(mpfYB, mpfXB, param['rd_dlf'])
    DLFscaleB = utl.MPF_DLFscale(mpfYB, mpfXB, param['rd_dlf'])
    maskA   = (DLFerrA <= param['th2_dlf']) & (DLFscaleA > param['th_scale'])
    maskB   = (DLFerrB <= param['th2_dlf']) & (DLFscaleB > param['th_scale'])
    if input_maskA is not None:
        maskA = maskA & input_maskA
    if input_maskB is not None:
        maskB = maskB & input_maskB
    
    if remove_mskA is not None:
        # removal of regions outside the panels
        maskA = maskA & remove_mskA
    
    if remove_mskB is not None:
        # removal of regions outside the panels
        maskB = maskB & remove_mskB
    
    # removal of regions outside the panels
    maskA = utl.removesmall(maskA, param['th_sizeA'], connectivity=8)
    maskB = utl.removesmall(maskB, param['th_sizeA'], connectivity=8)

    # removal regions without mirror
    mskB = utl.dilateDisk(maskB,param['rd_median'])
    mskA = utl.dilateDisk(maskA,param['rd_median'])
    mskAp, maskBp = utl.MPFmirror(mpfYA.astype(np.int32), mpfXA.astype(np.int32), maskA, mskB); 
    mskBp, maskAp = utl.MPFmirror(mpfYB.astype(np.int32), mpfXB.astype(np.int32), maskB, mskA); 
    maskA = (mskAp==1) | (maskAp==1)
    maskB = (mskBp==1) | (maskBp==1)
    
    # morphological operations
    maskA = utl.dilateDisk(maskA, param['rd_dil'])
    maskB = utl.dilateDisk(maskB, param['rd_dil'])
    maskA = binary_fill_holes(maskA)
    maskB = binary_fill_holes(maskB)
    
    return maskA, maskB
   
def get_colored(img, mask):
    img = img.copy()
    bord = utl.dilateDisk(mask,3) & (utl.erodeDisk(mask,3)==0)
    img[bord,0] = 1.0
    img[bord,1] = 0.0
    img[bord,2] = 0.0
        
    img = (255*img).clip(0,255).astype(np.uint8)
    return img