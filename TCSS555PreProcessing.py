from __future__ import print_function
import nltk
from nltk.stem import *
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from collections import defaultdict
from collections import Counter
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from string import ascii_lowercase
import spacy
import gensim.downloader as api
import os
import __future__
import re 
import numpy
import csv
import pandas as pd
import numpy as np
import math
import sys
import itertools

def removeSpecialCharacterToken(words):
    somelist = [x for x in words if not IsSpecialCharacterToken(x)]
    return somelist

def IsSpecialCharacterWord(word):
    symbols = ['`','~','!','@','#','$','%','^','&','*','(',')','_','-','+','=','{','[','}','}','|','\\',':',';','"','\'','<',',','>','.','?','/','""',' ','']
    for x in symbols:
        if word == x:
            return True
    return False

def IsSpecialCharacterToken(word):
    newWord = re.sub('[^A-Za-z0-9]+', '', word)
    if word == '':
        return True
    return False

def checkWords(wordArray):
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    for x in wordArray: 
        if x not in english_vocab:
            print(x)
    return words

def stripWords(words):
    i = 0
    for x in words:   
        newWord = str.strip(x)
        newWord = str.strip(newWord, "`~!@#$%^&*()_-+={[}}|\\:;\"<,>.?/ ")
        newWord = re.sub('[^A-Za-z0-9]+', '', newWord)
        newWord = newWord.lower()
        words[i] = newWord
        i = i + 1
    return words


def getWords(line):
    words = []
    for w in line.split():
        words.append(w)
    return words

def removeWords(df):
    textData = pd.DataFrame(list(df['text']))
    stop = stopwords.words('english')
    textData[0].replace('[!"#%\'()*+,-./:;<=>?@\[\]^_`{|}~1234567890’”“′‘\\\]',' ',inplace=True,regex=True)
    wordlist = filter(None, " ".join(list(set(list(itertools.chain(*textData[0].str.split(' ')))))).split(" "))
    df['stemmed_text_data'] = [' '.join(filter(None,filter(lambda word: word not in stop, line))) for line in textData[0].str.lower().str.split(' ')]
    minimum_count = 5
    str_frequencies = pd.DataFrame(list(Counter(filter(None,list(itertools.chain(*df['stemmed_text_data'].str.split(' '))))).items()),columns=['word','count'])
    low_frequency_words = set(str_frequencies[str_frequencies['count'] < minimum_count]['word'])
    return df


def Tokenize(words):
    words = word_tokenize(words);
    return words

def removeStopWords(words):
    stopWords = stopwords.words('english')
    somelist = [x for x in words if not IsStopWord(x,stopWords)]
    return somelist

def IsStopWord(word, stopWords):
    for x in stopWords:
        if word == x:
            return True
    return False

def removeNumbers(words):
    somelist = [x for x in words if not IsNumeric(x)]
    return somelist

def IsNumeric(word):
    if word.isnumeric():
        return True
    return False

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wn.ADJ,
                "N": wn.NOUN,
                "V": wn.VERB }

    return tag_dict.get(tag, wordnet.NOUN)

def lemmatizeWords(words):
    lemmatizer = WordNetLemmatizer()
    i = 0
    for x in words:           
        lemmizedWord = lemmatizer.lemmatize(x, get_wordnet_pos(x))
        words[i] = lemmizedWord
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

def processTextWithLemmanization(df, outputFilePath):
    lemmantizedData = []
    for index, row in df.iterrows(): 
        words = Tokenize(row['text'])
        words = stripWords(words)
        words = removeSpecialCharacterToken(words)
        words = removeStopWords(words)
        words = removeNumbers(words)
        lemmatizedWords = lemmatizeWords(words)
        newlemmatizedReview = " ".join(lemmatizedWords)
        d = {'id': row['id'], 'deceptive': row['deceptive'], 'hotel':row['hotel'], 'polarity':row['polarity'], 'source':row['source'], 'text':newlemmatizedReview }
        lemmantizedData.append(d)
    
    dfProcessedLemmatized = pd.DataFrame(data=lemmantizedData, columns=['id','deceptive','hotel','polarity','source','text'])
    dfProcessedLemmatized.to_csv(outputFilePath,index=False)


def processTextWithStemming(df, outputFilePath):
    stemmedData = []
    for index, row in df.iterrows(): 
        words = Tokenize(row['text'])
        words = stripWords(words)
        words = removeSpecialCharacterToken(words)
        words = removeStopWords(words)
        words = removeNumbers(words)
        stemmedWords = stemmPorterWords(words)
        newStemmedReview = " ".join(stemmedWords)
        d = {'id': row['id'], 'deceptive': row['deceptive'], 'hotel':row['hotel'], 'polarity':row['polarity'], 'source':row['source'], 'text':newStemmedReview }
        stemmedData.append(d)
    
    dfProcessedStemmed = pd.DataFrame(data=stemmedData, columns=['id','deceptive','hotel','polarity','source','text'])
    dfProcessedStemmed.to_csv(outputFilePath,index=False)

def processText(df, outputFilePath):
    data = []
    for index, row in df.iterrows(): 
        words = Tokenize(row['text'])
        words = stripWords(words)
        words = removeSpecialCharacterToken(words)
        words = removeStopWords(words)
        words = removeNumbers(words)
        words = " ".join(words)
        d = {'id': row['id'], 'deceptive': row['deceptive'], 'hotel':row['hotel'], 'polarity':row['polarity'], 'source':row['source'], 'text':words }
        data.append(d)
    
    dfProcessed = pd.DataFrame(data=data, columns=['id','deceptive','hotel','polarity','source','text'])
    dfProcessed.to_csv(outputFilePath,index=False)

