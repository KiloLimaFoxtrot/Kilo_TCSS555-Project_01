import os
import numpy
import csv
import pandas as pd
import numpy as np


def readTranscripts():
    rootdir = "C:\\Users\\cressm\\Desktop\\TCSS 555\\op_spam_v1.4\\"
    records = []
    id = 1
    for subdir, dirs, files in os.walk(rootdir):
         for file in files:
             subArray = subdir.split("\\")
             polarity = subArray[6].split("_")[0];
             deceptive = subArray[7].split("_")[0];
             source = subArray[7].split("_")[2];
             hotel = file.split("_")[1]
             fullFileName = subdir + "\\" +file
             with open(fullFileName, "r") as f:
                text = f.read()
                d = {'id': id, 'deceptive': deceptive, 'hotel':hotel, 'polarity':polarity,'source':source,'text': text}
                records.append(d)
             id = id + 1
    df = pd.DataFrame(data=records, columns=['id', 'deceptive','hotel','polarity','source','text'])
    return df



df = readTranscripts();
df.to_csv('C:\\Users\\cressm\\Desktop\\TCSS 555\\deceptive-opinion_indexed.csv',index=False)




