import os, sys, pdb
import numpy as np
from rVAD_fast import rVAD_fast
from data_loader import get_filepaths
from tqdm import tqdm


def createdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def VAD(wavpath):
    path_list = get_filepaths(wavpath, ".wav")
        
    for path in tqdm(path_list):
        
        wav_name = path.split("/")[-1]
        fvad = rVAD_fast(path) 
        np.savetxt(os.path.join(wavpath+'_VAD', wav_name.replace(".wav","")), fvad,  fmt='%i')         
      
if __name__=="__main__":
    wavpath = "/Users/yuwen/Desktop/quality_challenge/rVADfast_py_2.0/sample_wav"
    createdir(wavpath+'_VAD')
    VAD(wavpath)

     
