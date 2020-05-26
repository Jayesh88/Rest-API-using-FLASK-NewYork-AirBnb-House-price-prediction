# -*- coding: utf-8 -*-
"""
Created on Tue May 19 12:58:39 2020

@author: HP PC
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

rd_data=pd.read_csv("C:\\Users\\HP PC\\Downloads\\AB_NYC_2019.csv")

# total null valves
rd_data.isnull().sum()

#mean imputation for missing values in reviews per month section
rd_data['reviews_per_month'].fillna(rd_data['reviews_per_month'].mean(), inplace=True)

#converting character values to numerical
for i in range(0,len(rd_data)):
    if rd_data['room_type'][i]=='Private room':
        rd_data['room_type'][i]=1
    elif rd_data['room_type'][i]=='Entire home/apt':
        rd_data['room_type'][i]=2
    else:
        rd_data['room_type'][i]=3      

for i in range(0,len(rd_data)):
    if rd_data['neighbourhood_group'][i]=='Brooklyn':
        rd_data['neighbourhood_group'][i]=1
    else:
        rd_data['neighbourhood_group'][i]=2         
        
#droppping columns which have no impact on prediction such as name,id ,host id
housing_data=rd_data.drop(columns=['id','name','host_id','host_name','last_review','latitude','longitude','reviews_per_month','neighbourhood'])
housing_data=housing_data.dropna()


import seaborn as sns
f, ax = plt.subplots(figsize=(10, 8))
corr = housing_data.corr()
sns.heatmap(corr, annot=True, annot_kws={"size": 9}, cmap = sns.color_palette("PuOr_r", 50), 
                     vmin = -1, vmax = 1)


#creating dumy variables for Neighbourhood_group and Neighbourhood
#housing_data=pd.get_dummies(housing_data,columns=['neighbourhood_group'])
#housing_data=pd.get_dummies(housing_data,columns=['neighbourhood'])
#housing_data=pd.get_dummies(housing_data,columns=['room_type'])
housing_data.head()

housing_data.dtypes
housing_data=housing_data.apply(pd.to_numeric)

from sklearn.model_selection import train_test_split
y=housing_data['price']
x=housing_data.drop('price',axis=1)
X = x.apply(pd.to_numeric, errors='coerce')
Y = y.apply(pd.to_numeric, errors='coerce')

from sklearn.linear_model import LinearRegression as lm
xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.3, random_state = 42)
regressor=lm().fit(xTrain,yTrain)
predictions=regressor.predict(xTest)
#plt.scatter(predictions,yTest)

from sklearn.metrics import mean_squared_error, r2_score
#print("Coefficients(slope of the line):", model.coef_)
print("Mean squared error: %.2f"% mean_squared_error(predictions,yTest))
print("R-square: %.2f" % r2_score(yTest,predictions))

import pickle
# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
model.score(xTest,yTest)
print(model.predict([[2,9,6,4,5,6]]))


#from sklearn.ensemble import RandomForestRegressor
#rd=RandomForestRegressor(n_estimators=10,random_state=0)
#rd.fit(xTrain,yTrain)
#rd_predictions=rd.predict(xTest)
#
## Saving model to disk
#pickle.dump(regressor, open('model_random.pkl','wb'))
#
## Loading model to compare the results
#model_rnd = pickle.load(open('model_random.pkl','rb'))
#model_rnd.score(xTest,yTest)
#print(model_rnd.predict([[2,9,6,4,5,6]]))
#
#from flask import Flask, request, jsonify, render_template
#int_features = [int(x) for x in request.form.values()]
#final_features = [np.array(int_features)]
#prediction = model.predict([np.array(X)])


#predictions=model.predict(xTest)
#
#from sklearn.metrics import mean_squared_error, r2_score
#print("Coefficients(slope of the line):", model.coef_)
#print("Mean squared error: %.2f"% mean_squared_error(predictions,yTest))
#print("R-square: %.2f" % r2_score(yTest,predictions))