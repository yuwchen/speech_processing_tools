import os, sys, pdb
import numpy as np
from rVAD_fast import rVAD_fast
from tqdm import tqdm


def get_filepaths(directory):
      file_paths = []  
      for root, _, files in os.walk(directory):
            for filename in files:
                  filepath = os.path.join(root, filename)
                  if filename.endswith('.wav'):
                        file_paths.append(filepath)  
      return file_paths 


def createdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def VAD(wavpath):
    path_list = get_filepaths(wavpath)
        
    for path in tqdm(path_list):
        
        wav_name = path.split("/")[-1]
        fvad = rVAD_fast(path) 
        np.savetxt(os.path.join(wavpath+'_VAD', wav_name.replace(".wav","")), fvad,  fmt='%i')         
      
if __name__=="__main__":
    wavpath = "/Users/yuwen/Desktop/quality_challenge/rVADfast_py_2.0/sample_wav"
    createdir(wavpath+'_VAD')
    VAD(wavpath)

     
