import os
import whisper
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse


#Install whisper using: pip install -U openai-whisper
#https://github.com/openai/whisper

def get_filepaths(directory, format='.wav'):
      file_paths = []  
      for root, _, files in os.walk(directory):
            for filename in files:
                  if filename.endswith(format):
                        file_paths.append(filename)
      return file_paths 

def main():

      parser = argparse.ArgumentParser()
      parser.add_argument('--datadir', default='./datadir',  type=str, help='Path of your DATA/ directory')
      args = parser.parse_args()
      input_dir= args.datadir

      file_list = get_filepaths(input_dir, format='.wav') #loop all the .wav file in dir
      file_list = set(file_list)
      model = whisper.load_model("large")
      outputname = input_dir.split(os.sep)[-1]

      try:
            print('Number of files:', len(file_list))
            df = pd.read_csv(outputname+'_whisper_eng.csv')
            exist_list = set(df['wavname'].to_list())
            print('Number of already processed files:', len(exist_list))
            file_list = file_list - exist_list
            print('Number of unprocessed files:',len(file_list))
            file_list = list(file_list)
      except Exception as e:
            print(e)
            print('Create new file')
            df = pd.DataFrame(columns=['wavname','start', 'end', 'transcript'])
            df.to_csv(outputname+'_whisper_large.csv', sep=',', index=False, header=True)

      
      for filename in tqdm(file_list):
            path = os.path.join(input_dir, filename)
            result = model.transcribe(path, fp16=False)
            for seg in result['segments']:
                  transcript = seg['text']
                  start = seg['start']
                  end = seg['end']
                  results = pd.DataFrame([{'wavname':filename,'start':start,'end':end,'transcript':transcript}])
                  results.to_csv(outputname+'_whisper_large.csv', mode='a', sep=',', index=False, header=False)

if __name__ == '__main__':
    main()
