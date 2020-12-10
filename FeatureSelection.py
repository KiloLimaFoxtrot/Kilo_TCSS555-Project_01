import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

processed = pd.read_csv(r"C:\Users\cressm\Desktop\TCSS 555\deceptive-opinion_processed.csv",encoding='latin-1')
raw = pd.read_csv(r"C:\Users\cressm\Desktop\TCSS 555\deceptive-opinion.csv",encoding='latin-1')
punc = ['`','~','!','(',')','_','-','{','[','}','}',':',';','"',',','.','?','/','""']

processed['word_count'] = raw["text"].apply(lambda x: len(str(x).split(" ")))
processed['char_count'] = raw["text"].apply(lambda x: sum(len(word) for word in str(x).split(" ")))
processed['sentence_count'] = raw["text"].apply(lambda x: len(str(x).split(".")))
processed['avg_word_length'] = processed['char_count'] / processed['word_count']
processed['avg_sentence_length'] = processed['word_count'] / processed['sentence_count']
processed['word_density'] = processed['word_count'] / (processed['char_count'] + 1)
processed['punc_count'] = raw['text'].apply(lambda x : len([a for a in x if a in punc]))
processed['total_length'] = raw['text'].apply(len)
processed['capitals'] = raw['text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
#raw['caps_vs_length'] = raw.apply(lambda row: float(raw['capitals'])/float(raw['total_length']),axis=1)
processed['num_exclamation_marks'] = raw['text'].apply(lambda x: x.count('!'))
processed['num_question_marks'] = raw['text'].apply(lambda x: x.count('?'))
processed['num_punctuation'] = raw['text'].apply(lambda x: sum(x.count(w) for w in '.,;:'))
processed['num_symbols'] = raw['text'].apply(lambda x: sum(x.count(w) for w in '*&$%'))
processed['num_unique_words'] = raw['text'].apply(lambda x: len(set(w for w in x.split())))
processed['words_vs_unique'] = processed['num_unique_words'] / processed['word_count']
processed["word_unique_percent"] =  processed["num_unique_words"]*100/processed['word_count']

label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(processed['deceptive'])
hotelEncoded = label_encoder.fit_transform(processed['hotel'])
polarityEncoded = label_encoder.fit_transform(processed['polarity'])
sourceEncoded = label_encoder.fit_transform(processed['source'])

onehot_encoder = OneHotEncoder(sparse=False)
hotelEncoded = hotelEncoded.reshape(len(hotelEncoded), 1)
processed['hotel'] = onehot_encoder.fit_transform(hotelEncoded)
polarityEncoded = polarityEncoded.reshape(len(polarityEncoded), 1)
processed['polarity'] = onehot_encoder.fit_transform(polarityEncoded)
sourceEncoded = sourceEncoded.reshape(len(sourceEncoded), 1)
processed['source'] = onehot_encoder.fit_transform(sourceEncoded)


Tfidf_vect = TfidfVectorizer(max_features=3500)
Tfidf_vect.fit(processed['text'])
Train_X_Tfidf = Tfidf_vect.transform(processed['text'])
processed['text'] = Train_X_Tfidf.toarray()

X = processed.drop(['id','deceptive'], axis=1)

bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(19,'Score'))


model = ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()
