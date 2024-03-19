# Fish-Sound-Detection

## Audio Preprocessing
1. Gather all sound files and annotation txt file in one directory. (make sure each audio file could match a txt file having same file name)
   
2. Open raw_audio_process.py, change path & path2 to the directory in 1. Then set the csvname1&2 to somewhere for storing the results.
  
3. Open fish_sound_separation.py, change paths & csvnames, and generates two csv results, one consists of frames of fish sound, another consists of background sound(without fish sound).

## Training 
1. Open up train.py file in the folder of corresponding system (1D_CNN or 1D_LSTM).
2. Change names of csv files by editing AUDIO_0 and AUDIO_1 variables.
3. Run and wait.

## Testing 
1. Open up test.py file in the folder of corresponding system (1D_CNN or 1D_LSTM).
2. Change names of csv files by editing AUDIO_0 and AUDIO_1 variables.
3. Run and wait.

## Change Data Split
1. Open up load.py file in the folder of corresponding system (1D_CNN or 1D_LSTM).
2. Change the parameters in function prepro, current parameters:test_size=0.2, random_state=42
