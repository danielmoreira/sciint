#!/usr/bin/env python3

numThreads  = 4
cmd_rgb     = "python ./src/main_rgb.py '%s' '%s'"
cmd_zernike = "python ./src/main_zernike.py '%s' '%s'"
cmd_merge   = "python ./src/main_merge.py '%s' '%s' '%s' '%s'"

import os
import glob
import tqdm

folder_input  = os.path.join(os.getenv('CMFD_IO'), 'figures')
folder_output = os.path.join(os.getenv('CMFD_IO'), 'output_%s')
print('Input  folder:', folder_input)
print('Output folder:', folder_output%'merge')

def get_outputfile(inputfile, alg = 'merge'):
    return os.path.join(folder_output%alg, os.path.basename(inputfile)+'.npz')

listfile = sorted([x for x in glob.glob(folder_input+'/*.*g') if '_gt' not in x])
listfile = [ x for x in tqdm.tqdm(listfile) if not os.path.isfile(get_outputfile(x))]
print('Number of images:', len(listfile))

def funpar(inputfile):
    try:
        outputfile_rgb     = get_outputfile(inputfile, 'rgb')
        outputfile_zernike = get_outputfile(inputfile, 'zernike')
        outputfile_merge   = get_outputfile(inputfile)

        os.system(cmd_rgb % (inputfile, outputfile_rgb))
        os.system(cmd_zernike % (inputfile, outputfile_zernike))
        os.system(cmd_merge % (inputfile, outputfile_rgb, outputfile_zernike, outputfile_merge))

        return 0
    except:
        return -1

if numThreads<1:
    rets = [funpar(x) for x in tqdm.tqdm(listfile, total=len(listfile))]
else:
    from multiprocessing import get_context
    ctx  = get_context("fork")
    with ctx.Pool(numThreads) as pool:
        rets = list(tqdm.tqdm(pool.imap_unordered(funpar, listfile), total=len(listfile)))
