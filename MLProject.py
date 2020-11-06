
from __future__ import print_function
from nltk.stem import *
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer 
import nltk
import os
import __future__
import re 
import numpy
import csv
import pandas as pd
import numpy as np
import math
import sys

def getWords(line):
    words = []
    for w in line.split():
        words.append(w)
    return words

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
    somelist = [x for x in words if not IsStopWord(x,stopWords)]
    return somelist

def stemmSnowballWords(words):
    stemmer = SnowballStemmer("english")
    i = 0
    for x in words: 
        stemmedWord = stemmer.stem(x)
        words[i] = stemmedWord
        i = i + 1
    return words

def stemmPorterWords(words):
    stemmer = PorterStemmer()
    i = 0
    for x in words: 
        stemmedWord = stemmer.stem(x)
        words[i] = stemmedWord
        i = i + 1
    return words


def lemmatizeWords(words):
    lemmatizer = WordNetLemmatizer()
    i = 0
    for x in words:           
        lemmizedWord = lemmatizer.lemmatize(x)
        words[i] = lemmizedWord
        i = i + 1
    return words


def removeSpecialCharacterWords(words):
    somelist = [x for x in words if not IsSpecialCharacterWord(x)]
    return somelist

def IsSpecialCharacterWord(word):
    symbols = ['`','~','!','@','#','$','%','^','&','*','(',')','_','-','+','=','{','[','}','}','|','\\',':',';','"','\'','<',',','>','.','?','/','""',' ','']
    for x in symbols:
        if word == x:
            return True
    return False

def IsStopWord(word, stopWords):
    for x in stopWords:
        if word == x:
            return True
    return False


def stripWords(words):
    symbols = ['`','~','!','@','#','$','%','^','&','*','(',')','_','-','+','=','{','[','}','}','|','\\',':',';','"','\'','<',',','>','.','?','/','"',' ','']
    i = 0
    for x in words:       
        newWord = str.strip(x)
        newWord = str.strip(newWord, "`~!@#$%^&*()_-+={[}}|\\:;\"<,>.?/ ")
        newWord = re.sub('[^A-Za-z0-9]+', '', newWord)
        newWord = newWord.lower()
        words[i] = newWord
        i = i + 1
    return words

 
stopWords = getStopWords()

#set path for indexed decptive words
df = pd.read_csv('C:\\Users\\cressm\\Desktop\\TCSS 555\\deceptive-opinion_indexed.csv')  

lemmantizedData = []
stemmedData = []
for index, row in df.iterrows(): 
    words = getWords(row['text'])
    words = stripWords(words)
    words = removeSpecialCharacterWords(words)
    words = removeStopWords(words, stopWords)
    wordsToLemmative = words
    wordsToStem = words
    lemmatizedWords = lemmatizeWords(wordsToLemmative)
    #stemmedWords = stemmSnowballWords(wordsToStem)
    newlemmatizedReview = " ".join(lemmatizedWords)
    #newStemmedReview = " ".join(stemmedWords)
    #d = {'id': row['id'], 'deceptive': row['deceptive'], 'hotel':row['hotel'], 'polarity':row['polarity'], 'source':row['source'], 'text':newStemmedReview }
    d2 = {'id': row['id'], 'deceptive': row['deceptive'], 'hotel':row['hotel'], 'polarity':row['polarity'], 'source':row['source'], 'text':newlemmatizedReview }
    #stemmedData.append(d)
    lemmantizedData.append(d2)

#dfProcessed = pd.DataFrame(data=stemmedData, columns=['id','deceptive','hotel','polarity','source','text'])
dfProcessedLemmatized = pd.DataFrame(data=lemmantizedData, columns=['id','deceptive','hotel','polarity','source','text'])
#set path for output file
#dfProcessed.to_csv('C:\\Users\\cressm\\Desktop\\TCSS 555\\deceptive-opinion_processedstemmed.csv',index=False)
dfProcessedLemmatized.to_csv('C:\\Users\\cressm\\Desktop\\TCSS 555\\deceptive-opinion_processedlemmatized.csv',index=False)



