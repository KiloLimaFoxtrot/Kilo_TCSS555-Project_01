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

processed = pd.read_csv(r"C:\Users\cressm\Desktop\TCSS 555\deceptive-opinion_processed.csv",encoding='latin-1')
raw = pd.read_csv(r"C:\Users\cressm\Desktop\TCSS 555\deceptive-opinion.csv",encoding='latin-1')
punc = ['`','~','!','(',')','_','-','{','[','}','}',':',';','"',',','.','?','/','""']

raw['word_count'] = raw["text"].apply(lambda x: len(str(x).split(" ")))
raw['char_count'] = raw["text"].apply(lambda x: sum(len(word) for word in str(x).split(" ")))
raw['sentence_count'] = raw["text"].apply(lambda x: len(str(x).split(".")))
raw['avg_word_length'] = raw['char_count'] / raw['word_count']
raw['avg_sentence_lenght'] = raw['word_count'] / raw['sentence_count']
raw['word_density'] = raw['word_count'] / (raw['char_count'] + 1)
raw['punc_count'] = raw['text'].apply(lambda x : len([a for a in x if a in punc]))
raw['total_length'] = raw['text'].apply(len)
raw['capitals'] = raw['text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
#raw['caps_vs_length'] = raw.apply(lambda row: float(raw['capitals'])/float(raw['total_length']),axis=1)
raw['num_exclamation_marks'] = raw['text'].apply(lambda x: x.count('!'))
raw['num_question_marks'] = raw['text'].apply(lambda x: x.count('?'))
raw['num_punctuation'] = raw['text'].apply(lambda x: sum(x.count(w) for w in '.,;:'))
raw['num_symbols'] = raw['text'].apply(lambda x: sum(x.count(w) for w in '*&$%'))
raw['num_unique_words'] = raw['text'].apply(lambda x: len(set(w for w in x.split())))
raw['words_vs_unique'] = raw['num_unique_words'] / raw['word_count']
raw["word_unique_percent"] =  raw["num_unique_words"]*100/raw['word_count']


