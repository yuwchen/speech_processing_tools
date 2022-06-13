import librosa
import math
import os
import numpy as np
import random
import csv
from scipy.io import wavfile

def get_filepaths(directory):
      file_paths = []  
      for root, directories, files in os.walk(directory):
            for filename in files:
                  # Join the two strings in order to form the full filepath.
                  filepath = os.path.join(root, filename)
                  if filename.endswith('.wav'):
                        file_paths.append(filepath)  
      return file_paths  


def creat_dir(directory):
      if not os.path.exists(directory):
            os.makedirs(directory)

def add_noise(clean_wav_path, noise_wav_path, SNR, return_info=False):
      
      #read clean wav
      y_clean, clean_rate = librosa.load(clean_wav_path, sr=16000)
      y_clean = y_clean.astype('float64')     
      noise_ori, noise_rate = librosa.load(noise_wav_path, sr=16000)
      noise_ori = noise_ori.astype('float64')   

      #if noise longer than clean wav
      if len(noise_ori) < len(y_clean):
            tmp = (len(y_clean) // len(noise_ori)) + 1
            y_noise = []
            for _ in range(tmp):
                  y_noise.extend(noise_ori)
      else:
            y_noise = noise_ori

      # cut noise
     
      start = random.randint(0,len(y_noise)-len(y_clean))
      end = start+len(y_clean)
      y_noise = y_noise[start:end]     
      y_noise = np.asarray(y_noise)

      y_clean_pw = np.dot(y_clean,y_clean) 
      y_noise_pw = np.dot(y_noise,y_noise) 


      noise = np.sqrt(y_clean_pw/((10.0**(SNR/10.0))*y_noise_pw))*y_noise
      y_noisy = y_clean + noise
      y_noisy = y_noisy/np.max(abs(y_noisy))

      P_c = np.dot(y_clean, y_clean) #/ len(y_clean)
      P_n = np.dot(noise, noise) #/ len(noise)
      print("Check SNR:",10*math.log10((P_c/P_n)))

      if return_info is False:
            return y_noisy, clean_rate
      else:
            info = {}
            info['start'] = start
            info['end'] = end
            return y_noisy, clean_rate, info


noise_path = "/path/to/noise/dir"
clean_path = "/path/to/clean/wav"
Noisy_path = "/path/to/output/wav"

noise_list = get_filepaths(noise_path)
clean_list = get_filepaths(clean_path)


SNR_list = [-10,-7,-4,-1,1,4,7,10] 
num_of_copy = 2 #randomly select how many types of noise

for snr in SNR_list:

      with open(Noisy_path+Noisy_path.split("/")[-2]+"_"+str(snr)+'.csv', 'w', newline='') as csvFile:
            fieldNames = ['wav','noise', 'start','end']
            writer = csv.DictWriter(csvFile, fieldNames)
            writer.writeheader()
            for clean_wav_path in clean_list:

                  noise_wav_path_list = random.sample(noise_list, num_of_copy)
                  #noise_wav_path_list = noise_list #mix with all noise in the dir

                  for noise_wav_path in noise_wav_path_list:
                        y_noisy, clean_rate, info = add_noise(clean_wav_path, noise_wav_path, snr, True)
                        noise_name = noise_wav_path.split(os.sep)[-1].split(".")[0]
                        output_dir = Noisy_path+os.sep+str(snr)+os.sep+noise_name
                        creat_dir(output_dir)
                        wav_name = clean_wav_path.split(os.sep)[-1]
                        librosa.output.write_wav( output_dir+os.sep+wav_name, y_noisy, clean_rate) 
                        writer.writerow({'wav':wav_name,'noise':noise_name, 'start':info['start'], 'end':info['end']})



