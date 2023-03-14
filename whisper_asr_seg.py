import os
import whisper
import numpy as np
import pandas as pd
from tqdm import tqdm

#Install whisper using: pip install -U openai-whisper
#https://github.com/openai/whisper

def get_filepaths(directory, format='.wav'):
      file_paths = []  
      for root, _, files in os.walk(directory):
            for filename in files:
                  filepath = os.path.join(root, filename)
                  if filename.endswith(format):
                        file_paths.append(filepath)  
      return file_paths 

input_dir = '/path/to/wav/dir'
file_list = get_filepaths(input_dir, format='.wav') #loop all the .wav file in dir

df =  pd.DataFrame(columns=['wavname','transcript'])

model = whisper.load_model("base")

for path in tqdm(file_list):

      filename = os.path.basename(path)
      result = model.transcribe(path, fp16=False)
      for seg in result['segments']:
            transcript = seg['text']
            start = seg['start']
            end = seg['end']
            results = pd.DataFrame([{'wavname':filename,'start':start,'end':end,'transcript':transcript}])
            df = pd.concat([df, results])

outputname = input_dir.split(os.sep)[-1]
df.to_csv(outputname+'_whisper.csv', sep=',', index=False)




