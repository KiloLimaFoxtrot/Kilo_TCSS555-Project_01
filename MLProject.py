
from __future__ import print_function
from nltk.stem import *
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
import nltk
import os
import __future__
import re 
import numpy
import csv
import pandas as pd
import numpy as np
import math


def getWords(line):
    words = []
    for w in line.split():
        words.append(w)
    df = pd.DataFrame(data=words, columns=['word'])
    return df


def printDataFrameText(words):
    wordArray = words.to_numpy().ravel()
    newReview = " ".join(wordArray)
    print(newReview)

def getStopWords():
    #Set path for location of stop word file
    stopwordsFile = "C:\\Users\\cressm\\Desktop\\TCSS 554\\Assignment1\\stopwords.txt"
    stopwords = []
    with open(stopwordsFile, "r") as f:
       for line in f:
         stopwords.extend(line.split())
    return stopwords

def removeStopWords(words, stopWords):
    words = words.drop(words[words['word'].isin(stopWords)].index)
    return words

def stemmSnowballWords(words):
    stemmer = SnowballStemmer("english")
    for index, row in words.iterrows():
        if index < len(words.index):
            stemmedWord = stemmer.stem(row['word'])
            words.iat[index,0] = stemmedWord
    return words

def stemmPorterWords(words):
    stemmer = PorterStemmer()
    for index, row in words.iterrows():
        if index < len(words.index):
            stemmedWord = stemmer.stem(row['word'])
            words.iat[index,0] = stemmedWord
    return words

def removeSpecialCharacterWords(words):
    symbols = ['`','~','!','@','#','$','%','^','&','*','(',')','_','-','+','=','{','[','}','}','|','\\',':',';','"','\'','<',',','>','.','?','/','""',' ','']
    words = words.drop(words[words['word'].isin(symbols)].index)
    return words

def stripWords(words):
    symbols = ['`','~','!','@','#','$','%','^','&','*','(',')','_','-','+','=','{','[','}','}','|','\\',':',';','"','\'','<',',','>','.','?','/','"',' ','']
    for index, row in words.iterrows():       
        newWord = str.strip(row['word'])
        newWord = str.strip(newWord, "`~!@#$%^&*()_-+={[}}|\\:;\"<,>.?/ ")
        newWord = re.sub('[^A-Za-z0-9]+', '', newWord)
        newWord = newWord.lower()
        words.iat[index,0] = newWord
    return words

 
stopWords = getStopWords()

#set path for indexed decptive words
df = pd.read_csv('C:\\Users\\cressm\\Desktop\\TCSS 555\\deceptive-opinion_indexed.csv')  

data = []
for index, row in df.iterrows(): 
    words = getWords(row['text'])
    words = stripWords(words)
    words = removeSpecialCharacterWords(words)
    words = removeStopWords(words, stopWords)
    words = stemmSnowballWords(words)
    wordArray = words.to_numpy().ravel()
    newReview = " ".join(wordArray)
    d = {'id': row['id'], 'deceptive': row['deceptive'], 'hotel':row['hotel'], 'polarity':row['polarity'], 'source':row['source'], 'text':newReview }
    data.append(d)

dfProcessed = pd.DataFrame(data=data, columns=['id','deceptive','hotel','polarity','source','text'])
#set path for output file
dfProcessed.to_csv('C:\\Users\\cressm\\Desktop\\TCSS 555\\deceptive-opinion_processed.csv',index=False)



