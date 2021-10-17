#!/usr/bin/env python

import os
import glob
import tqdm
import pandas
import numpy as np
from PIL import Image

folder_input  = os.path.join(os.getenv('CMFD_IO'), 'figures')
folder_gt     = os.path.join(os.getenv('CMFD_IO'), 'gt')
folder_output = os.path.join(os.getenv('CMFD_IO'), 'output_%s')
print('Input  folder:', folder_input)
print('GT     folder:', folder_gt)
print('Output folder:', folder_output%'merge')

def get_gtfile(inputfile):
    return os.path.join(folder_gt, os.path.splitext(os.path.basename(inputfile))[0]+'.png')

def get_outputfile(inputfile, alg = 'merge'):
    return os.path.join(folder_output%alg, os.path.basename(inputfile)+'.npz')

def get_f1(gt, mask):
    TP = np.float64(np.sum(np.logical_and(gt,mask)))
    FP = np.float64(np.sum(np.logical_and(np.logical_not(gt),mask)))
    FN = np.float64(np.sum(np.logical_and(gt,np.logical_not(mask))))
    f1 = 2*TP/(2*TP+FP+FN)
    return f1

listfile = sorted([x for x in glob.glob(folder_input+'/*.*g') if '_gt' not in x])
listfile = [ x for x in tqdm.tqdm(listfile) if os.path.isfile(get_outputfile(x)) and os.path.isfile(get_gtfile(x))]


tab = {
    'filename': list(),
    'time_rgb': list(),
    'time_zernike': list(),
    'time_merge': list(),
    'f1_rgb': list(),
    'f1_zernike': list(),
    'f1_merge': list(),  
}

for inputfile in tqdm.tqdm(listfile, total=len(listfile)):
    filename = os.path.basename(inputfile)
    gtfile             = get_gtfile(inputfile)
    outputfile_rgb     = get_outputfile(inputfile, 'rgb')
    outputfile_zernike = get_outputfile(inputfile, 'zernike')
    outputfile_merge   = get_outputfile(inputfile)

    gt = np.asarray(Image.open(gtfile))<128
    
    time_rgb     = np.load(outputfile_rgb)['time']
    time_zernike = np.load(outputfile_zernike)['time']
    time_merge   = np.load(outputfile_merge)['time']
    f1_rgb     = get_f1(gt,np.load(outputfile_rgb)['map'])
    f1_zernike = get_f1(gt,np.load(outputfile_zernike)['map'])
    f1_merge   = get_f1(gt,np.load(outputfile_merge)['map'])
    
    tab['filename'].append(filename)
    tab['time_rgb'].append(time_rgb)
    tab['time_zernike'].append(time_zernike)
    tab['time_merge'].append(time_merge)
    tab['f1_rgb'].append(f1_rgb)
    tab['f1_zernike'].append(f1_zernike)
    tab['f1_merge'].append(f1_merge)

tab = pandas.DataFrame(tab)
tab.to_csv(os.path.join(os.getenv('CMFD_IO'), 'results.csv'))

print('', flush=True)
print('Number of images:', len(tab))
print('****************************************')
print('|        |   RGB   | Zernike |  Merge  |')
print('|F1 mean |  %5.3f  |  %5.3f  |  %5.3f  |'% (np.mean(tab['f1_rgb']), np.mean(tab['f1_zernike']), np.mean(tab['f1_merge']) ))
print('|F1 std  |  %4.2f   |  %4.2f   |  %4.2f   |'% (np.std(tab['f1_rgb']), np.std(tab['f1_zernike']), np.std(tab['f1_merge']) ))
print('****************************************')
