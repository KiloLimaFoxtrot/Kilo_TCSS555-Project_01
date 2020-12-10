import pandas as pd
import numpy as np
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression


Corpus = pd.read_csv(r"C:\Users\cressm\Desktop\TCSS 555\deceptive-opinion_processed.csv",encoding='latin-1')

y = Corpus['deceptive']
X = Corpus.drop(['id','deceptive'], axis=1)

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(X,y,test_size=0.3)


label_encoder = LabelEncoder()
Train_Y = label_encoder.fit_transform(Train_Y)
Test_Y = label_encoder.fit_transform(Test_Y)
hotelEncoded_Train = label_encoder.fit_transform(Train_X['hotel'])
polarityEncoded_Train = label_encoder.fit_transform(Train_X['polarity'])
sourceEncoded_Train = label_encoder.fit_transform(Train_X['source'])
hotelEncoded_Test = label_encoder.fit_transform(Test_X['hotel'])
polarityEncoded_Test = label_encoder.fit_transform(Test_X['polarity'])
sourceEncoded_Test = label_encoder.fit_transform(Test_X['source'])


onehot_encoder = OneHotEncoder(sparse=False)
hotelEncoded_Train = hotelEncoded_Train.reshape(len(hotelEncoded_Train), 1)
Train_X['hotel'] = onehot_encoder.fit_transform(hotelEncoded_Train)
HotelArray_Train = onehot_encoder.fit_transform(hotelEncoded_Train)

polarityEncoded_Train = polarityEncoded_Train.reshape(len(polarityEncoded_Train), 1)
Train_X['polarity'] = onehot_encoder.fit_transform(polarityEncoded_Train)
#PolarityArray_Train = onehot_encoder.fit_transform(PolarityArray_Train)

sourceEncoded_Train = sourceEncoded_Train.reshape(len(sourceEncoded_Train), 1)
Train_X['source'] = onehot_encoder.fit_transform(sourceEncoded_Train)

hotelEncoded_Test = hotelEncoded_Test.reshape(len(hotelEncoded_Test), 1)
Test_X['hotel'] = onehot_encoder.fit_transform(hotelEncoded_Test)
HotelArray_Test= onehot_encoder.fit_transform(hotelEncoded_Test)

polarityEncoded_Test = polarityEncoded_Test.reshape(len(polarityEncoded_Test), 1)
#PolarityArray_Test = onehot_encoder.fit_transform(polarityEncoded_Test)
Test_X['polarity'] = onehot_encoder.fit_transform(polarityEncoded_Test)

sourceEncoded_Test = sourceEncoded_Test.reshape(len(sourceEncoded_Test), 1)
Test_X['source'] = onehot_encoder.fit_transform(sourceEncoded_Test)

Tfidf_vect = TfidfVectorizer(max_features=3500)
Tfidf_vect.fit(Corpus['text'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X['text'])
Test_X_Tfidf = Tfidf_vect.transform(Test_X['text'])
Train_X['text'] = Train_X_Tfidf.toarray()
Test_X['text'] = Test_X_Tfidf.toarray()
TrainX_Array = Train_X_Tfidf.toarray()
TestX_Array = Test_X_Tfidf.toarray()


model = LogisticRegression(C=1.0).fit(HotelArray_Train , Train_Y)
score = model.score(HotelArray_Test, Test_Y)
print('features', score)

np.array([Test_X.columns[1:-1]]).T
feature_importance=pd.DataFrame(np.hstack((np.array([Train_X['hotel'].columns]).T, model.coef_.T)), columns=['feature', 'importance'])
feature_importance['importance'] = pd.to_numeric(feature_importance['importance'])
feature_importance = feature_importance.sort_values(by='importance', ascending=False)
print(feature_importance)


