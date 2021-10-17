import numpy as np
try:
    from scipy.misc import factorial
except:
    from scipy.special import factorial
from scipy.signal import convolve
from scipy.signal import fftconvolve


def FiltersBank(x, bfdata, mode='valid'):
    number = bfdata['number']
    bf = bfdata['filters']
    wf = bfdata['factors']
    dtype = np.result_type(x.dtype,bf.dtype)
    if mode=='valid':
        y = np.zeros([x.shape[0]-bf.shape[0]+1,x.shape[1]-bf.shape[1]+1, number], dtype=dtype)
    else:
        y = np.zeros([x.shape[0], x.shape[1], number], dtype=dtype)
    for index in range(number):
        y[:,:,index] = wf[index]*convolve(x, bf[:,:,index], mode=mode)
    return y


def FiltersBank_FFT(x, bfdata, mode='valid'):
    number = bfdata['number']
    bf = bfdata['filters']
    wf = bfdata['factors']
    dtype = np.result_type(x.dtype,bf.dtype)
    if mode=='valid':
        y = np.zeros([x.shape[0]-bf.shape[0]+1,x.shape[1]-bf.shape[1]+1, number], dtype=dtype)
    else:
        y = np.zeros([x.shape[0], x.shape[1], number], dtype=dtype)
    for index in range(number):
        y[:,:,index] = wf[index]*fftconvolve(x, bf[:,:,index], mode=mode)
    return y


def ZM_orderlist(ORDER):
    NM = list()
    for n  in range(ORDER+1):
        for m in range(n+1):
            if ((n-abs(m))%2)==0:
                NM = NM + [(n,m),]
    return NM


def ZM_bf(SZ = 16, ORDER = 5, dtype = np.float32):

    if np.isscalar(ORDER):
        NM  = ZM_orderlist(ORDER)
    else:
        NM  = ORDER

    num = len(NM)
    ORDER = np.max(np.asarray(NM)[:, 0])

    BFr = np.zeros((SZ, SZ, num), dtype=dtype)
    BFi = np.zeros((SZ, SZ, num), dtype=dtype)
    WF  = np.zeros((num,), dtype=dtype)

    F = factorial( np.arange(ORDER+1) )

    [X,Y] = np.meshgrid(np.arange(SZ),np.arange(SZ))
    rho   = np.sqrt(    (2.0*Y-SZ+1.0)**2+(2.0*X-SZ+1.0)**2)/SZ
    theta = np.arctan2(-(2.0*Y-SZ+1.0),   (2.0*X-SZ+1.0))

    mask  = (rho<=1)
    cnt   = np.sum(mask)
    for index in range(num):
        n = NM[index][0]
        m = NM[index][1]

        Rad = np.zeros_like(rho, dtype=dtype)
        tu = int((n+abs(m))/2)
        td = int((n-abs(m))/2)
        for s in range(td+1):
            c = ( ((-1.0)**s) * F[n-s] ) / (F[s]*F[tu-s]*F[td-s])
            Rad = Rad + c * (rho**(n - 2.0*s))


        BFr[:, :, index] = mask * Rad * np.cos(m * theta)
        BFi[:, :, index] = mask * Rad * np.sin(m * theta)
        WF[index]        = np.sqrt((n+1.0)/cnt)


    bfdata = dict()
    bfdata['number' ] = num
    bfdata['name'   ] = 'Zernike'
    bfdata['orders' ] = NM
    bfdata['filters'] = (BFr + 1j *BFi)
    bfdata['factors'] = WF
    return bfdata


def ZMp_bf(SZ=16, ORDER=5, radiusNum=26, anglesNum=32, dtype = np.float32):

    if np.isscalar(ORDER):
        NM  = ZM_orderlist(ORDER)
    else:
        NM  = ORDER

    num = len(NM)
    ORDER = np.max(np.asarray(NM)[:, 0])

    BFr = np.zeros((SZ, SZ, num), dtype=dtype)
    BFi = np.zeros((SZ, SZ, num), dtype=dtype)
    WF = np.zeros((num,), dtype=dtype)
    coef = np.zeros((num, ORDER), dtype=dtype)

    F = factorial(np.arange(ORDER + 1))


    for index in range(num):
        n = NM[index][0]
        m = NM[index][1]
        tu = int((n+abs(m))/2)
        td = int((n-abs(m))/2)

        for s in range(td+1):
            coef[index, s] = (((-1.0) ** s) * F[n - s]) / (F[s] * F[tu - s] * F[td - s])

        WF[index] = np.sqrt(n+1.0)


    gf = lambda x,y: np.maximum(0.0,1.0-np.abs(x)) * np.maximum(0.0,1.0-np.abs(y)) # bi-linear

    [X,Y] = np.meshgrid(np.arange(SZ),np.arange(SZ))
    Y = (2.0 *Y-SZ+1.0)/2.0
    X = (2.0 *X-SZ+1.0)/2.0

    radiusMax = (SZ-1.0)/2.0
    radiusMin = 0
    radiusPoints = np.linspace(radiusMin,radiusMax,radiusNum)


    for indexA in range(anglesNum):
        A = np.deg2rad(indexA*360.0/anglesNum)

        for indexR in range(radiusNum):
            R = radiusPoints[indexR]
            pX = X - R*np.cos(A)
            pY = Y - R*np.sin(A)
            J  = gf(pX, pY)
            Rn = 2.0*R / SZ
            for index in range(num):
                n = NM[index][0]
                m = NM[index][1]
                Rad = 0
                td = int((n-abs(m))/2)
                for s in range(td+1):
                    Rad = Rad + coef[index,s]  * (Rn**(n-2.0*s))

                BFr[:, :, index] = BFr[:, :, index] +  (J*Rad)*Rn * np.cos(m*A)
                BFi[:, :, index] = BFi[:, :, index] +  (J*Rad)*Rn * np.sin(m*A)


    WF = WF / np.sqrt(np.sum((BFr[:,:,0])**2+(BFi[:,:,0])**2))

    bfdata = dict()
    bfdata['number'] = num
    bfdata['name'] = 'ZernikePolar'
    bfdata['orders'] = NM
    bfdata['filters'] = (BFr + 1j * BFi)
    bfdata['factors'] = WF
    return bfdata


def PCT_bf(SZ=16, ORDER=3, dtype = np.float32):
    if np.isscalar(ORDER):
        [n, m] = np.meshgrid(range(ORDER), range(ORDER))
        n = n.flatten()
        m = m.flatten()
        NM  = [(n[index],m[index]) for index in range(n.size)]
    else:
        NM  = ORDER

    num = len(NM)
    ORDER = np.max(np.asarray(NM)[:, 0])

    BFr = np.zeros((SZ, SZ, num), dtype=dtype)
    BFi = np.zeros((SZ, SZ, num), dtype=dtype)
    WF = np.zeros((num,), dtype=dtype)


    [X,Y] = np.meshgrid(np.arange(SZ),np.arange(SZ))
    rho2  =           ( (2.0*Y-SZ+1.0)**2+(2.0*X-SZ+1.0)**2)/(SZ**2)
    theta = np.arctan2(-(2.0*Y-SZ+1.0),   (2.0*X-SZ+1.0))
    mask  = (rho2<=1)
    cnt   = np.sum(mask)
    for index in range(num):
        n = NM[index][0]
        m = NM[index][1]

        Rad = np.cos((np.pi)*n*rho2)
        BFr[:, :, index] = mask * Rad * np.cos(m * theta)
        BFi[:, :, index] = mask * Rad * np.sin(m * theta)
        WF[index] = ((n>0)+1.0)/cnt

    bfdata = dict()
    bfdata['number'] = num
    bfdata['name'] = 'PCT'
    bfdata['orders'] = NM
    bfdata['filters'] = (BFr + 1j * BFi)
    bfdata['factors'] = WF
    return bfdata


def PCTp_bf(SZ=16, ORDER=3, radiusNum=26, anglesNum=32, dtype = np.float32):
    if np.isscalar(ORDER):
        [n, m] = np.meshgrid(range(ORDER), range(ORDER))
        n = n.flatten()
        m = m.flatten()
        NM  = [(n[index],m[index]) for index in range(n.size)]
    else:
        NM  = ORDER

    num = len(NM)
    ORDER = np.max(np.asarray(NM)[:, 0])

    BFr = np.zeros((SZ, SZ, num), dtype=dtype)
    BFi = np.zeros((SZ, SZ, num), dtype=dtype)
    WF = np.zeros((num,), dtype=dtype)

    for index in range(num): WF[index] = ((NM[index][0]>0)+1)

    gf = lambda x, y: np.maximum(0.0, 1.0 - np.abs(x)) * np.maximum(0.0, 1.0 - np.abs(y))  # bi-linear

    [X,Y] = np.meshgrid(np.arange(SZ),np.arange(SZ))
    Y = (2.0 *Y-SZ+1.0)/2.0
    X = (2.0 *X-SZ+1.0)/2.0

    radiusMax = (SZ-1.0)/2.0
    radiusMin = 0
    radiusPoints = np.linspace(radiusMin,radiusMax,radiusNum)

    for indexA in range(anglesNum):
        A = np.deg2rad(indexA*360.0/anglesNum)

        for indexR in range(radiusNum):
            R = radiusPoints[indexR]
            pX = X - R*np.cos(A)
            pY = Y - R*np.sin(A)
            J  = gf(pX, pY)
            Rn = 2.0*R / SZ
            for index in range(num):
                n = NM[index][0]
                m = NM[index][1]
                Rad = np.cos((np.pi)*n*Rn*Rn)
                BFr[:, :, index] = BFr[:, :, index] + (J * Rad) * Rn * np.cos(m * A)
                BFi[:, :, index] = BFi[:, :, index] + (J * Rad) * Rn * np.sin(m * A)

    WF = WF / np.sqrt(np.sum((BFr[:,:,0])**2+(BFi[:,:,0])**2))

    bfdata = dict()
    bfdata['number'] = num
    bfdata['name'] = 'PCTpolar'
    bfdata['orders'] = NM
    bfdata['filters'] = (BFr + 1j * BFi)
    bfdata['factors'] = WF
    return bfdata


def FMTpl_bf(SZ=24, freq_m = range(5), radiusNum=26, anglesNum=32, freq_n = range(26), radiusMin = np.sqrt(2), dtype = np.float32):
    num_freq_m = len(freq_m)
    num_freq_n = len(freq_n)
    num = num_freq_m*num_freq_n

    [n, m] = np.meshgrid(freq_n, freq_m)
    NM = [(n.flat[index], m.flat[index]) for index in range(n.size)]
    freq_n = [(index % radiusNum) for index in freq_n]


    gf = lambda x, y: np.maximum(0.0, 1.0 - np.abs(x)) * np.maximum(0.0, 1.0 - np.abs(y))  # bi-linear

    [X,Y] = np.meshgrid(np.arange(SZ),np.arange(SZ))
    Y = (2.0 *Y-SZ+1.0)/2.0
    X = (2.0 *X-SZ+1.0)/2.0

    radiusMax = (SZ-1.0)/2.0
    radiusPoints = (2.0 ** np.linspace(np.log2(radiusMin),np.log2(radiusMax),radiusNum))

    BF = np.zeros((SZ,SZ,anglesNum,radiusNum), dtype=dtype)
    for indexA in range(anglesNum):
        A = np.deg2rad(indexA * 360.0 / anglesNum)

        for indexR in range(radiusNum):
            R = radiusPoints[indexR]
            pX = X - R * np.cos(A)
            pY = Y - R * np.sin(A)
            J = gf(pX, pY)
            BF[:,:,indexA,indexR] = J


    BF = np.fft.fft(BF, axis=2)/anglesNum
    BF = np.fft.fft(BF, axis=3)
    BF = BF[:,:,freq_m,:]
    BF = BF[:,:,:,freq_n]

    WF = np.ones((num,), dtype=dtype)
    BF = BF.reshape((SZ,SZ,num))

    bfdata = dict()
    bfdata['number'] = num
    bfdata['name'] = 'FMTlogpolar'
    bfdata['orders'] = NM
    bfdata['filters'] = BF
    bfdata['factors'] = WF
    bfdata['radiusStep'] = np.log(radiusPoints[-1]) - np.log(radiusPoints[-2])
    bfdata['freq0'] = 1.0 / (radiusNum * bfdata['radiusStep'])
    return bfdata


