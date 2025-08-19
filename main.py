#telechargement des bibliotheques necessaires

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix


#loading the dataset

data = pd.read_csv('customers-100.csv')
X = data['City']
y = data['Country']
print(X)

#splitting the data

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)