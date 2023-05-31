import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import cross_val_score,KFold,StratifiedKFold

from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

import pickle
import asyncio

# 1. Problem Statement
# Here 50 startups dataset containing 5 columns like
# “R&D Spend”, “Administration”, “Marketing Spend”, “State”, “Profit”. In this dataset the first 3 columns
# provide you spending on Research , Administration and Marketing respectively. State indicates startup
# based on that state. Profit indicates how much profits earned by a startup.
# Prepare a prediction model for profit of 50_Startups data in Python
# Dependant variable : “R&D Spend”, “Administration”, “Marketing Spend”, “State”
# Independant variable : “Profit”

# Data Gathering
profit_df=pd.read_csv(r'/home/agasti/Desktop/Assignment 6/app/data.csv')
print(profit_df.head())
print(profit_df.columns)
print(profit_df.info())

#Exploratory Data Analysis
print(profit_df.isna().sum())
print(profit_df['R&D Spend'])
print(profit_df.describe())
print(profit_df.boxplot('R&D Spend'))
# plt.show()
print(profit_df['R&D Spend'].skew())

print(profit_df.isna().sum())
print(profit_df['Administration'])
print(profit_df.boxplot('Administration'))
# plt.show()
print(profit_df['Administration'].skew())

print(profit_df.isna().sum())
print(profit_df['Marketing Spend'])
print(profit_df.boxplot('Marketing Spend'))
# plt.show()
print(profit_df['Marketing Spend'].skew())

print(profit_df.isna().sum())
print(profit_df['State'])

profit_df['State'] = profit_df['State'].replace({'New York':0,'California':1,'Florida':2})

print(profit_df.corr().tail())
print(sns.heatmap(profit_df.corr().tail(),annot=True))
# plt.show()

for i in ['R&D Spend', 'Administration', 'Marketing Spend', 'State']:
    print(sns.scatterplot(profit_df,x=i,y='Profit'))
    # plt.show()

from statsmodels.stats.outliers_influence import variance_inflation_factor

x = profit_df.drop('Profit',axis=1)

# print(variance_inflation_factor(x.values,1))

vif_list = []
for i in range(x.shape[1]):
    vif = variance_inflation_factor(x.values,i)
    vif_list.append(round(vif,2))

s1 = pd.Series(vif_list,index=x.columns)
plt.figure(figsize=(20,20))
print(s1.sort_values().plot(kind='barh'))
# plt.show()

# Train Test Split
x = profit_df.drop('Profit',axis=1)
y = profit_df['Profit']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

def model_building(algo,x,y):
    model = algo
    model.fit(x,y)
    return model

linear_reg = model_building(LinearRegression(),x_train,y_train)
print(linear_reg)

def evaluation(model,ind_var,y_act):
    pred=model.predict(ind_var)

    mse=mean_squared_error(y_act,pred)
    print('MSE : ',mse)

    mae=mean_absolute_error(y_act,pred)
    print('MAE : ',mae)

    r2_squared = r2_score(y_act,pred)
    print('R2_Score : ',r2_squared)

print('Test Data Evaluation'.center(50,'*'))
evaluation(linear_reg,x_test,y_test)

print('Train Data Evaluation'.center(50,'*'))
evaluation(linear_reg,x_train,y_train)

# print(linear_reg.predict([[2000,3000,1000,0]]))

def save_model():
    # Code to save the model
    model_path = r'/home/agasti/Desktop/Assignment 6/app/model/model_pkl'
    pickle.dump(linear_reg, open(model_path, 'wb'))
    return model_path

def load_model():
    # Load the saved model from the 'model' directory
    pickled_model = pickle.load(open(save_model(), 'rb'))
    return pickled_model


