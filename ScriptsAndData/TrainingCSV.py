import pandas as pd
import numpy as np

from math import floor

## See all available files
import os
import shutil

r_directory = "GeneratedData/"
w_directory = "TrainingData/Ready/"
c_directory = "TrainingData/Test/"

available_files = os.listdir(r_directory)
av_num = len (available_files)

train_split = floor(av_num * 0.6)
eval_split  = floor(av_num * 0.2) + train_split

print ("Files Separation: ")
print (".. Train: {}, Eval: {}, Test: {}".format(train_split, eval_split - train_split, av_num - (eval_split)) )
print (".. Total Data: {}".format(av_num) )

ind = np.arange( av_num )
np.random.shuffle(ind)

filename = "train_data"
wmode = 'w'
hd = True

print ("..start writing files")
for i in range (av_num):
    

    if (i == train_split):
        filename = "eval_data"
        wmode = 'w'
        hd = True
    elif (i == eval_split):
        filename = "test_data"
        wmode = 'w'
        hd = True

    ourDF = pd.read_csv("{}{}".format(r_directory, available_files[ind[i]]))
    ourDF.to_csv( "{}{}.{}".format(w_directory, filename, 'csv'), index=False, mode=wmode, header=hd )

    if filename == "test_data":
        shutil.copy("{}{}".format(r_directory, available_files[ind[i]]), "{}{}".format(c_directory, available_files[ind[i]]))

    wmode = 'a'
    hd = False

print ("..Making Training CSV Done")
