import numpy
import scipy.ndimage as simg

def MPFspacedist2(mpfY, mpfX):
    [pX, pY] = numpy.meshgrid(numpy.arange(mpfY.shape[1]), numpy.arange(mpfY.shape[0]))
    dist2    = numpy.abs(mpfY-pY)**2 + numpy.abs(mpfX-pX)**2
    return dist2

def genDisk(radius):
    [X, Y] = numpy.meshgrid(numpy.arange(-radius, radius + 1), numpy.arange(-radius, radius + 1))
    D = ((X**2+Y**2)<=(radius**2))
    return D

def MPFregularize(mpfY, mpfX, radius, mode='reflect'):
    [pX, pY] = numpy.meshgrid(numpy.arange(mpfY.shape[1]), numpy.arange(mpfY.shape[0]))
    D = genDisk(radius)
    mpfY = simg.median_filter(mpfY - pY, footprint = D, mode = mode) + pY
    mpfX = simg.median_filter(mpfX - pX, footprint = D, mode = mode) + pX
    return mpfY, mpfX

def DLFaffineMtx(radius):
    [X, Y] = numpy.meshgrid(numpy.arange(-radius, radius + 1), numpy.arange(-radius, radius + 1))
    D = ((X**2+Y**2)<=(radius**2)).astype(numpy.float64)
    d = D.reshape(1,-1)
    M = numpy.matmul(d.transpose(),d)
    x = numpy.concatenate((d*Y.flatten(), d*X.flatten(), d), 0)
    B = numpy.linalg.pinv(x)
    A = M * numpy.matmul(B, x)
    A = (A + A.transpose()) / 2.0
    [S, V] = numpy.linalg.eigh(A)
    V = V[:, numpy.where(S > 0.5)]
    V = V.reshape((2 * radius + 1, 2 * radius + 1, -1))
    return D, V

def DLFestimeMtx(radius):
    [X, Y] = numpy.meshgrid(numpy.arange(-radius, radius + 1), numpy.arange(-radius, radius + 1))
    Z = ((X**2+Y**2)<=(radius**2)).astype(numpy.float64)
    x = numpy.stack((Z*Y, Z*X, Z), -1).reshape((-1,3), order='C')
    xp = numpy.linalg.pinv(x)
    xp = xp.reshape((-1, 2 * radius + 1, 2 * radius + 1), order='C')*Z[None,...]
    return xp[0], xp[1], xp[2]

def Derror(x, D, mode = 'reflect'):
    y = numpy.abs(simg.correlate(x, D, mode = mode))
    return y

def DLFerror(x, D, V, mode = 'reflect'):
    y = simg.correlate(x ** 2, D, mode = mode) - (
        simg.correlate(x, V[:,:,0], mode = mode) ** 2 +
        simg.correlate(x, V[:,:,1], mode = mode) ** 2 +
        simg.correlate(x, V[:,:,2], mode = mode) ** 2 )
    return y

def MPF_DCFerror(mpfY, mpfX, D, mode='reflect'):
    [pX, pY] = numpy.meshgrid(numpy.arange(mpfY.shape[1]), numpy.arange(mpfY.shape[0]))
    mpfY = mpfY - pY
    mpfX = mpfX - pX
    e = simg.correlate(mpfY ** 2, D, mode = mode) - (simg.correlate(mpfY, D, mode = mode) ** 2) + \
        simg.correlate(mpfX ** 2, D, mode = mode) - (simg.correlate(mpfX, D, mode = mode) ** 2)
    return e


def MPF_DLFerror(mpfY, mpfX, radius, mode = 'reflect'):
    D, V = DLFaffineMtx(radius)
    e = DLFerror(mpfY, D, V, mode) + DLFerror(mpfX, D, V, mode)
    return e

def MPF_DLFscale(mpfY, mpfX, radius, mode = 'reflect'):
    fY, fX, fZ = DLFestimeMtx(radius)
    s = simg.correlate(mpfY, fY, mode = mode) * simg.correlate(mpfX, fX, mode = mode) - \
        simg.correlate(mpfX, fY, mode = mode) * simg.correlate(mpfY, fX, mode = mode)
    return numpy.abs(s)


def MPFdual(mpfY, mpfX, mask):
    Nr = mask.shape[0]
    Nc = mask.shape[1]

    ind0,ind1 = numpy.where(mask)
    for index in range(len(ind0)):
        py = int(mpfY[ind0[index], ind1[index]])
        px = int(mpfX[ind0[index], ind1[index]])
        if (py>=0) & (py<Nr) & (px>=0) & (px<Nc):
            mask[py,px] = True
    return mask

def MPFmirror(mpfY, mpfX, mask, val):
    Nr = val.shape[0]
    Nc = val.shape[1]
    outA = numpy.nan*numpy.ones( list(mask.shape) + list(val.shape[2:]))
    outB = numpy.nan*numpy.ones( val.shape)
    ind0,ind1 = numpy.where(mask>0)
    for index in range(len(ind0)):
        py = int(mpfY[ind0[index], ind1[index]])
        px = int(mpfX[ind0[index], ind1[index]])
        if (py>=0) & (py<Nr) & (px>=0) & (px<Nc):
            outA[ind0[index], ind1[index]] = val[py,px]
            outB[py,px] = val[py,px]
    return outA, outB


def removesmall(x, size, connectivity=8):
    y, mapsnum = simg.label(x, simg.generate_binary_structure(x.ndim, connectivity))
    x = numpy.zeros_like(x, dtype=numpy.bool)
    for index in range(1,mapsnum+1):
        map = (y==index)
        mapsize = numpy.sum(map)
        if mapsize>=size:
            x = x | map
    return x

def dilateDisk(x, radius):
    return simg.binary_dilation(x, structure=genDisk(radius))

def erodeDisk(x, radius):
    return simg.binary_erosion(x, structure=genDisk(radius))

def closeDisk(x, radius):
    return simg.binary_closing(x, structure=genDisk(radius))

se_def  = numpy.asarray([[0, 1, 0], [ 1, 1, 1], [ 0, 1, 0]], dtype=bool)

def hysteresis(maskH, maskL, se = se_def):
    maskH = maskH.astype(bool)
    maskL = maskL.astype(bool)

    mask = simg.binary_dilation(maskH, structure=se) & maskL
    while not (numpy.array_equal(mask,maskH)):
        maskH = mask
        mask  = simg.binary_dilation(maskH, structure=se) & maskL

    return mask