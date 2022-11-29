

rVADfast 2.0 python code.   

- run_rVAD.py  

  run the VAD. the code will loop all the '.wav' files in directory and calucalte the VAD results of each wavfile  
  the results will be saved as the same name in the path/to/wavfiles/dir_VAD directory  
  input: path/to/wavfiles/dir   
  output: VAD results (0 indicates no speech, whereas 1 indicates speech)   
  
  
- waveform_segmentation.py  
  use the VAD detection result to segment the wavform   
  input: (1) path/to/wavfiles/dir (2) path/to/wavfiles/dir_VAD (3) path/to/output/file  
  output: segmentation results  
  the segmentation will be named as 'originalFileName'\_'startSamplePoint'\_'endSamplePoint'.wav 
