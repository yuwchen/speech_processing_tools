from __future__ import division
import numpy as np
from scipy.signal import lfilter
import speechproc
from copy import deepcopy

# Refs:
#  [1] Z.-H. Tan, A.k. Sarkara and N. Dehak, "rVAD: an unsupervised segment-based robust voice activity detection method," Computer Speech and Language, 2019. 
#  [2] Z.-H. Tan and B. Lindberg, "Low-complexity variable frame rate analysis for speech recognition and voice activity detection." 
#  IEEE Journal of Selected Topics in Signal Processing, vol. 4, no. 5, pp. 798-807, 2010.

# 2017-12-02, Achintya Kumar Sarkar and Zheng-Hua Tan
# Usage: python rVAD_fast_2.0.py inWaveFile  outputVadLabel

def rVAD_fast(finwav):
    
    fs, data = speechproc.speech_wave(finwav) 
    
    if data.ndim==1:# signal-channel
        vad = calculate_vad(fs, data)
    elif data.ndim==2: # two-channel
        vad_1 = calculate_vad(fs, data[:,0])
        vad_2 = calculate_vad(fs, data[:,1])
        assert len(vad_1)==len(vad_2)
        vad_1 = np.asarray(vad_1)
        vad_2 = np.asarray(vad_2)
        vad = vad_1 & vad_2
    else:
        print("Not supported")

    return vad

def calculate_vad(fs, data):

    winlen, ovrlen, _, _, nftt = 0.025, 0.01, 0.97, 20, 512
    ftThres=0.5; vadThres=0.4
    opts=1

    ft, flen, fsh10, nfr10 =speechproc.sflux(data, fs, winlen, ovrlen, nftt)
    # --spectral flatness --

    pv01= np.zeros(nfr10)
    tmp = np.less_equal(ft, ftThres)
    try:    
        pv01[tmp]=1 
    except Exception as e:
        tmp = np.append(tmp, False)
        pv01[tmp]=1 
    pitch=deepcopy(ft)

    pvblk=speechproc.pitchblockdetect(pv01, pitch, nfr10, opts)

    # --filtering--
    ENERGYFLOOR = np.exp(-50)
    b=np.array([0.9770,   -0.9770])
    a=np.array([1.0000,   -0.9540])
    fdata=lfilter(b, a, data, axis=0)

    #--pass 1--
    noise_samp, _, n_noise_samp=speechproc.snre_highenergy(fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk)

    #sets noisy segments to zero
    for j in range(n_noise_samp):
        fdata[range(int(noise_samp[j,0]),  int(noise_samp[j,1]) +1)] = 0 

    vad_seg=speechproc.snre_vad(fdata,  nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk, vadThres)
    vad = vad_seg.astype(int)
    data=None; pv01=None; pitch=None; fdata=None; pvblk=None; vad_seg=None 

    return vad 