import numpy as np
from PIL import Image

def imread(strFile):
    img = Image.open(strFile)
    img = np.asarray(img)

    return img

def imread2f(strFile, channel = 1, dtype = np.float32):
    img = Image.open(strFile)
    if (channel==3):
        img = img.convert('RGB')
        img = np.asarray(img).astype(dtype) / 256.0
    elif (channel==1):
        if img.mode == 'L':
            img = np.asarray(img).astype(dtype) / 256.0
        else:
            img = img.convert('RGB')
            img = np.asarray(img).astype(dtype)
            img = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2])/256.0
    else:
        img = np.asarray(img).astype(dtype) / 256.0

    return img

def img2grayf(img, dtype = np.float32):
    in_dtype = img.dtype
    img = img.astype(dtype)

    if in_dtype == np.uint8:
        img = img/256.0
    elif in_dtype == np.uint16:
        img = img/65536.0
    elif in_dtype == np.uint32:
        img = img/(2.0**32)
    elif in_dtype == np.uint64:
        img = img/(2.0**64)


    if (img.ndim == 3):
        if (img.shape[2] == 3):
            img = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2])

    if img.ndim>2: img = np.mean(img, axis=2)

    return img
