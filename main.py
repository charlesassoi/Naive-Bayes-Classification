#telechargement des bibliotheques necessaires

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.6,random_state=42)

#text processing:converting text to numeric features
vectorizer=CountVectorizer()
X_train_vectorized=vectorizer.fit_transform(X_train)
X_test_vectorized=vectorizer.transform(X_test)

#training Naive Bayes Classification

model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

#making prediction
y_pred=model.predict(X_test_vectorized)

#evaluating the model
accuracy=accuracy_score(y_test,y_pred)
conf_matrix=confusion_matrix(y_test,y_pred)
print(f'accuracy{accuracy*100}%')
class_labels = np.unique(y_test)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
