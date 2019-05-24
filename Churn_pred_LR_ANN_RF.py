# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:12:01 2019

@author: iasedric
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('P12-Churn-Modelling.csv')
y = dataset.iloc[:, 13].values


X = dataset.iloc[:, [3,6,7,8,9,10,11,12]]

# Taking care of categorical variables
categories = ['France','Spain','Germany']
cate_country= pd.get_dummies(dataset.iloc[:,4], columns=categories)
cate_country = cate_country.astype('float64')

categories = ['Female','Male']
cate_gender= pd.get_dummies(dataset.iloc[:,5], columns=categories)
cate_gender = cate_gender.astype('float64')

X = pd.concat([X, cate_country, cate_gender], axis=1, sort=False).values



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy',max_depth = 2 , random_state = 0)


#Applying K-fold Cross Validation
#from sklearn.model_selection import cross_val_score
#accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

#mean = accuracies.mean()
#std = accuracies.std()

#Applying the Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators' : [90,92,94,96,98,100,102,104,106,108,110],
                'max_depth' : [8,9,10,11,12]
                }]

grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters, 
                           scoring = 'accuracy',
                           cv = 10, 
                           n_jobs = -1)


grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


# Fitting Random Forest Classification to the Training set

classifier = RandomForestClassifier(n_estimators = best_parameters['n_estimators'], criterion = 'entropy',max_depth = best_parameters['max_depth'] , random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred_Test_proba = classifier.predict_proba(X_test)

y_pred_Train_proba = classifier.predict_proba(X_train)

CAP = np.stack((y_train,y_pred_Train_proba[:,1]), axis=1)

CAP = pd.DataFrame({'y_train' : CAP[:,0],
                    'y_train_proba' : CAP[:,1]}).sort_values(by=['y_train_proba'], ascending =False).to_csv(r"C:\Users\iasedric.REDMOND\Documents\_Perso\Training\Data Science A-Z Template Folder\Churn\For_CAP_wRegressionForest.csv", index=False, encoding='utf_8_sig')

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_RF = confusion_matrix(y_test, y_pred)


CAP = np.stack((y_test,y_pred_Test_proba[:,1]), axis=1)

CAP = pd.DataFrame({'y_test' : CAP[:,0],
                    'y_test_proba' : CAP[:,1]}).sort_values(by=['y_test_proba'], ascending =False).to_csv(r"C:\Users\iasedric.REDMOND\Documents\_Perso\Training\Data Science A-Z Template Folder\Churn\For_CAP_wRegressionForest_test.csv", index=False, encoding='utf_8_sig')





