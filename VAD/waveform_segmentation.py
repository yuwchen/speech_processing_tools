
import os
from scipy.io import wavfile
import pickle
from tqdm import tqdm
import pandas as pd
import soundfile as sf


def createdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_filepaths(directory):
      file_paths = []  
      for root, _, files in os.walk(directory):
            for filename in files:
                  filepath = os.path.join(root, filename)
                  if filename.endswith('.wav'):
                        file_paths.append(filepath)  
      return file_paths 


def get_boundary(vad):
    boundary = []
    previous_flag = '0'
    for i in range(len(vad)):
        current_flag = vad[i]
        if current_flag!=previous_flag:
            boundary.append(i)
            previous_flag = current_flag
    return boundary

rootdir = './sample_wav' #path to original wavfiles
VADdir = './sample_wav_VAD' #path to VAD results 
outputdir = rootdir+'_seg' #output segmentation results
createdir(outputdir)

wav_list = get_filepaths(rootdir)

for path in wav_list:

    filename = os.path.basename(path).replace(".wav","")
    sample_rate, wav = wavfile.read(path)
    vad = open(os.path.join(VADdir,filename),"r").read().splitlines()
    boundary = get_boundary(vad)
    
    scale = wav.shape[0]/len(vad)
    for i in range(len(boundary)//2):
        b_idx = int(i*2)
        seg_length = int(boundary[b_idx+1]*scale)-int(boundary[b_idx]*scale)
        if seg_length<=8000: #skip the segmentant that is too short
            continue
        else:
            start = int(boundary[b_idx]*scale)
            end = int(boundary[b_idx+1]*scale)
            the_wav = wav[start:end]
            sf.write(os.path.join(outputdir,filename+'_'+str(start)+'_'+str(end)+'.wav'), the_wav, sample_rate)
    