# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 15:19:20 2023

@author: zameer
"""

#------------------Company Data---------------#

#Import the Data
import pandas as pd
import numpy as np
df=pd.read_csv('Company_Data.csv')
df
df.head()
df.shape
list(df)
df.dtypes

#Data Transformation
df['Sales'].mean()

df['Sales']=np.select(
   [df['Sales']<df['Sales'].mean(),
    df['Sales']>df['Sales'].mean()],
   ['low_sales','high_sales'],
   default='Other')

df.head()
list(df)

#label encoding
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['ShelveLoc'] =LE.fit_transform(df['ShelveLoc'])
df['Urban']=LE.fit_transform(df['Urban'])
df['US']=LE.fit_transform(df['US'])
df['Sales']=LE.fit_transform(df['Sales'])
df

#splitting as X and Y
Y=df["Sales"]
X=df.iloc[:,1:10]

#standardization
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(X)
SS_X = pd.DataFrame(SS_X)
SS_X


#Data partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y,test_size=0.30)

#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=100,max_depth=8,
                             max_samples=0.6,
                             max_features=0.7)

from sklearn.metrics import accuracy_score
RFC.fit(X_train,Y_train)

Y_pred_train = RFC.predict(X_train)
Y_pred_test = RFC.predict(X_test)

training_accuracy = []
test_accuracy =[]

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y,test_size=0.30,random_state=i)
    RFC.fit(X_train,Y_train)
    Y_pred_train = RFC.predict(X_train)
    Y_pred_test = RFC.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))
    
k1 = pd.DataFrame(training_accuracy)
k2 = pd.DataFrame(test_accuracy)


print("Cross validation accuracy results:",k1.mean().round(2))
print("Cross validation test accuracy results",k2.mean().round(2))
print("variance",np.mean(k1.mean()-k2.mean()).round(2))

#================FRAUD CHECK DATA================#

#Import the data
import pandas as pd
import numpy as np
df=pd.read_csv("Fraud_check.csv")
df
df.head()
df.shape
df.dtypes
list(df)

#Defining Taxable.Income as Risky nd good
#Define the ranges and coressponding categories
low_range = (0,30000)
high_range = (30001,float('inf'))

#create a new categorical variable based on the defined ranges
df['Income_category'] = np.select(
    [df['Taxable.Income'].between(*low_range),df['Taxable.Income'].between(*high_range)],
    ['Risky','Good'],
    default='Other')

df.head()
list(df)

# Data Transformation
#label encoding
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['Undergrad'] = LE.fit_transform(df['Undergrad'])
df['Marital.Status'] = LE.fit_transform(df['Marital.Status'])
df['Urban'] = LE.fit_transform(df['Urban'])
df['Income_category'] = LE.fit_transform(df['Income_category'])

#splitting X and Y
Y=df["Income_category"]
X=df.iloc[:,[3,4]]

# Standardization
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(X)
SS_X = pd.DataFrame(SS_X)
SS_X

#Data partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y,test_size=0.30)

#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=100,max_depth=8,
                             max_samples=0.6,
                             max_features=0.7)

from sklearn.metrics import accuracy_score
RFC.fit(X_train,Y_train)

Y_pred_train = RFC.predict(X_train)
Y_pred_test = RFC.predict(X_test)

training_accuracy = []
test_accuracy =[]

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(SS_X,Y,test_size=0.30,random_state=i)
    RFC.fit(X_train,Y_train)
    Y_pred_train = RFC.predict(X_train)
    Y_pred_test = RFC.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train,Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test,Y_pred_test))
    
k1 = pd.DataFrame(training_accuracy)
k2 = pd.DataFrame(test_accuracy)


print("Cross validation accuracy results:",k1.mean().round(2))
print("Cross validation test accuracy results",k2.mean().round(2))
print("variance",np.mean(k1.mean()-k2.mean()).round(2))












































