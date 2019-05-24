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



# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy',max_depth = 6 , random_state = 0)
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


#Applying K-fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

mean = accuracies.mean()
std = accuracies.std()


CAP = np.stack((y_test,y_pred_Test_proba[:,1]), axis=1)

CAP = pd.DataFrame({'y_test' : CAP[:,0],
                    'y_test_proba' : CAP[:,1]}).sort_values(by=['y_test_proba'], ascending =False).to_csv(r"C:\Users\iasedric.REDMOND\Documents\_Perso\Training\Data Science A-Z Template Folder\Churn\For_CAP_wRegressionForest_test.csv", index=False, encoding='utf_8_sig')


# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:15:54 2019

@author: iasedric
"""


"""
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred_Test_proba = classifier.predict_proba(X_test)

y_pred_Train_proba = classifier.predict_proba(X_train)

CAP = np.stack((y_train,y_pred_Train_proba[:,1]), axis=1)

CAP = pd.DataFrame({'y_train' : CAP[:,0],
                    'y_train_proba' : CAP[:,1]}).sort_values(by=['y_train_proba'], ascending =False).to_csv(r"C:\Users\iasedric.REDMOND\Documents\_Perso\Training\Data Science A-Z Template Folder\Churn\For_CAP_wLogisticRegression.csv", index=False, encoding='utf_8_sig')



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_LR = confusion_matrix(y_test, y_pred)

CAP = np.stack((y_test,y_pred_Test_proba[:,1]), axis=1)

CAP = pd.DataFrame({'y_test' : CAP[:,0],
                    'y_test_proba' : CAP[:,1]}).sort_values(by=['y_test_proba'], ascending =False).to_csv(r"C:\Users\iasedric.REDMOND\Documents\_Perso\Training\Data Science A-Z Template Folder\Churn\For_CAP_wLogisticRegression_test.csv", index=False, encoding='utf_8_sig')



# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 13))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred_Test_proba = classifier.predict(X_test)
y_pred = (y_pred_Test_proba > 0.5)

y_pred_Train_proba = classifier.predict(X_train)


#CAP = np.stack((y_train,y_pred_Train_proba[:,1]), axis=1)

CAP = np.stack((y_train,y_pred_Train_proba[:,0]), axis=1)

CAP = pd.DataFrame({'y_train' : CAP[:,0],
                    'y_train_proba' : CAP[:,1]}).sort_values(by=['y_train_proba'], ascending =False).to_csv(r"C:\Users\iasedric.REDMOND\Documents\_Perso\Training\Data Science A-Z Template Folder\Churn\For_CAP_wANN.csv", index=False, encoding='utf_8_sig')


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_ANN = confusion_matrix(y_test, y_pred)


CAP = np.stack((y_test,y_pred_Test_proba[:,0]), axis=1)

CAP = pd.DataFrame({'y_test' : CAP[:,0],
                    'y_test_proba' : CAP[:,1]}).sort_values(by=['y_test_proba'], ascending =False).to_csv(r"C:\Users\iasedric.REDMOND\Documents\_Perso\Training\Data Science A-Z Template Folder\Churn\For_CAP_wANN_test.csv", index=False, encoding='utf_8_sig')


"""
