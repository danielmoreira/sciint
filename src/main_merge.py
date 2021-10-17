import os
from PIL import Image
import numpy as np
import sys
sys.path.append(os.path.dirname(__file__))

from utility.utilityImage import imread2f, img2grayf
from core import get_colored

if __name__=="__main__":
    from time import time
    if len(sys.argv)==5:
        filename_img = sys.argv[1]  # input image file name
        filename_in1 = sys.argv[2]  # 1st input npz file name
        filename_in2 = sys.argv[3]  # 2nd input npz file name
        filename_out = sys.argv[4]  # output image file name
        filename_out, filename_ext = os.path.splitext(filename_out)
        if filename_ext=='.npz': filename_ext = '.png'
        filename_out_dat = filename_out + '.npz'
        filename_out_msk = filename_out + filename_ext
        filename_out_col = filename_out + '_col' + filename_ext
        
        dat1 = dict(np.load(filename_in1))
        dat2 = dict(np.load(filename_in2))
        timestamp = time()
        img = imread2f(filename_img, channel=3)
        mask = np.logical_or(dat1['map'], dat2['map'])
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
        outmat['time'] = timeApproach + dat1['time'] + dat2['time']
        try:
            os.makedirs(os.path.dirname(filename_out_dat), exist_ok=True)
        except:
            pass
        np.savez(filename_out_dat, **outmat)
        Image.fromarray(colored_output).save(filename_out_col)
        mask = 255 * np.logical_not(mask).astype(np.uint8)
        Image.fromarray(mask).save(filename_out_msk)
        
    else:
        print('number of pparameters is not correct')