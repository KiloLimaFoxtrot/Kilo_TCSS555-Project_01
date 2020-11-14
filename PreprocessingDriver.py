import TCSS555PreProcessing as lib
import pandas as pd


df = pd.read_csv('C:\\Users\\cressm\\Desktop\\TCSS 555\\deceptive-opinion.csv')  
lib.processTextWithLemmanization(df, 'C:\\Users\\cressm\\Desktop\\TCSS 555\\deceptive-opinion_test1.csv')




