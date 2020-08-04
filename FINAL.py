# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 23:35:38 2020

@author: Pratik Bagellu
"""
#Import 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
import pandas as pd
import re 
import nltk

#CODE
train= pd.read_csv('train.csv')
Master=Master.dropna()
Master=Master.reset_index(drop=True)
dataset  = pd.read_csv('Finalcleaned.csv',sep = ',')
dataset= dataset[0:8331]
test  = pd.read_csv('test.csv',sep = ',')
tweetprocessed = [ ]
dataset = dataset.dropna()
dataset['location']
#dataset.getdummies()
#Master
dataset = pd.concat([train,test],axis=0)
dataset = dataset.reset_index(drop='true')
#Processing for Training Dataset.
for i in range(0,10876):
        
    dataset['text'][i] =dataset['text'][i].split()
    dataset['text'][i] =' '.join(word for word in dataset['text'][i] if not word.startswith('http'))
            
    dataset['text'][i]= re.sub('[^a-zA-Z]',' ',dataset['text'][i] )
    dataset['text'][i] = dataset['text'][i].lower()
    dataset['text'][i] = dataset['text'][i].split() 
    ps=PorterStemmer()
    dataset['text'][i] = [ps.stem(word) for word in dataset['text'][i] if not word in set(stopwords.words('english'))]
    dataset['text'][i] = ' '.join(dataset['text'][i])
    #tweetprocessed.append(dataset['text'][i])
    # Cleaning location and
        
    dataset['keyword'][i] =dataset['keyword'][i].split()
    dataset['keyword'][i] =''.join(word for word in dataset['keyword'][i] if not word.startswith('http'))
            
    dataset['keyword'][i]= re.sub('[^a-zA-Z]',' ',dataset['keyword'][i] )
    dataset['keyword'][i] = dataset['keyword'][i].lower()
    dataset['keyword'][i] = dataset['keyword'][i].split() 
    ps=PorterStemmer()
    dataset['keyword'][i] = [ps.stem(word) for word in dataset['keyword'][i] if not word in set(stopwords.words('english'))]
    dataset['keyword'][i] = ''.join(dataset['keyword'][i])
        #tweetprocessed.append(dataset['keyword'][i])


# Bag of words 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X= cv.fit_transform(dataset['text']).toarray()
X1= cv.fit_transform(dataset['keyword']).toarray()
combined= np.concatenate((X,X1), axis=1)
X_train= X[0:7613,:]
y_train= dataset['target'][0:7613]
X_test= X[7613:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#using Random forest Best accracy 
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators= 10, criterion= 'entropy')
classifier.fit(X_train, y_train)

# Create Decision Tree classifer object
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
classifier= DecisionTreeClassifier()
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
pd.concat([master['text'][0:7612])
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm=confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)

#Submission Data
Submit= dataset[7612:]
Submit=Submit.drop(columns=['target'])
Submit= Submit.reset_index(drop=True)
Submit=pd.concat([Submit,pd.DataFrame(y_pred)],axis=1)
Finalsubmission= Submit.drop([Submit.columns[1],Submit.columns[2],Submit.columns[3]],axis=1)

#Exporting Submission data to csv 
Finalsubmission.to_csv('Result.csv')
pd.concat([Submit,y_pred],axis=1)
