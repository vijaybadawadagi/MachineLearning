# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 15:52:46 2018

@author: VijayB
"""
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

seed = 7
dataframe = pandas.read_csv('iris.csv',header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)

def baseline_model():
    #create model
    model = Sequential()
    model.add(Dense(8,input_dim=4,activation='relu'))
    model.add(Dense(3,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model,epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True,random_state=seed)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print('Baseline: %.2f%% (%.2f%%)' % (results.mean()*100, results.std()*100))

estimator.fit(X,Y)
predictions = estimator.predict(Y)
print('Predictions',predictions)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(X,predictions)
print ("\nThe confusion matrix when apply the test set on the trained nerual network:\n" , cm)