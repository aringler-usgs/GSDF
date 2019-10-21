#!/usr/bin/env python
import math
import numpy as np

def filtergauss(data, f0):
    # f0 is the central frequency
    # alpha is our spread
    ld = len(data)
    band = 2.5
    alpha = 3./band**2
    nfft = int(2**(math.ceil(math.log(len(data),2))))
    data = np.fft.rfft(data, n=nfft)
    w = 2.*np.pi*np.fft.rfftfreq(nfft, 1)
    w0=2.*np.pi*f0
    H=np.zeros(len(w)) 
    for idx in range(len(w)):
        if (w[idx] >= (1-band)*w0) and (w[idx] <= (1+band)*w0):
            H[idx] =np.exp(-alpha*(w[idx]-w0)**2/(w0**2))
    data *= H
    data[-1] = abs(data[-1]) + 0.0j
    data = np.fft.irfft(data)[0:ld]
    return data

def taper_fun(data, f0):
    idx1 = int(len(data)/2 - (1./f0)/2.)
    data[:idx1] *= 0.
    idx2 = int(len(data)/2 + (1./f0)/2.)
    data[idx2:] *= 0.
    data[idx1:idx2] *= np.hanning(idx2-idx1)
    return data
        
def wavelet(A, sigma, tg, w, tp, t):
    return A*np.exp(-0.5*((t-tg)*sigma)**2)*np.cos(w*(t-tp))
