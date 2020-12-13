# Sources:
# 1. https://machinelearningmastery.com/feature-selection-machine
# -learning-python/
# 2. https://machinelearningmastery.com/calculate-feature-importance
# -with-python/
# 3. https://towardsdatascience.com/machine-learning-simple-linear
# -regression-with-python-f04ecfdadc13

# **** THIS CLASS IS NOT FUNCTIONAL

import pandas as pd
from numpy import set_printoptions
from pandas import read_csv
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import TCSS555PreProcessing as lib

# import networkx.drawing.tests.test_pylab

output_path = 'C:\\Users\\cressm\\Desktop\\TCSS 555\\deceptive-opinion.csv'

df = pd.read_csv('../deceptive-opinion.csv')
lib.processTextWithLemmanization(df,
                                 output_path +
                                 'deceptive-opinion_test1.csv')

# load data
filename = 'deceptive-opinion_test1.csv'
names = ['id', 'deceptive', 'hotel', 'polarity', 'source', 'text']
dataframe = read_csv(filename, names=names)
array1 = dataframe.values
X = array1[:, 0:5]
Y = array1[:, 5]
# feature extraction
model = LogisticRegression(solver='LRSolver')
rfe1 = RFE(model, 3) # not sure about 3

### Fitting the data has been the holdup.
# returns the ValueError: could not convert string to float: 'id'
dataFit1 = rfe1.fit(X, Y)
print("The Number of Features: %d" % dataFit1.n_features_)
print("The Number of Features: %d" % dataFit1.n_features_)
print("The Selected Features: %s" % dataFit1.support_)
print("The Feature Ranking: %s" % dataFit1.ranking_)

test = SelectKBest(score_func=f_classif, k=4)
dataFit1 = test.fit(X, Y)
# summarize scores
set_printoptions(precision=3)
print(dataFit1.scores_)
features = dataFit1.transform(X)
# summarize selected features
print(features[0:5, :])
