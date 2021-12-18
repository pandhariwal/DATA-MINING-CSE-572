#!/usr/bin/env python
# coding: utf-8

# In[86]:


import pandas as pd
import pandas
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import pywt
from sklearn.model_selection import train_test_split
import scipy
import scipy.fftpack
from numpy.fft import fft
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import pickle

insulinDf = pandas.read_csv('InsulinData.csv',parse_dates=[['Date','Time']],low_memory=False,usecols=['Date','Time','BWZ Carb Input (grams)']).iloc[::-1]
cgmDf = pandas.read_csv('CGMData.csv',parse_dates=[['Date','Time']],low_memory=False,usecols=['Date','Time','Sensor Glucose (mg/dL)']).iloc[::-1]

insulinDfset2 = pandas.read_csv('InsulinAndMealIntake670GPatient3.csv',parse_dates=[['Date','Time']],low_memory=False,usecols=['Date','Time','BWZ Carb Input (grams)']).iloc[::-1]
cgmDfset2 = pandas.read_csv('CGMData670GPatient3.csv',parse_dates=[['Date','Time']],low_memory=False,usecols=['Date','Time','Sensor Glucose (mg/dL)']).iloc[::-1]


# In[87]:


def makeDf(insulinDF, cgmDf):
    mealDataMatrix = []
    noMealDataMatrix = []

    insulinneutralize = insulinDf[
        insulinDf["BWZ Carb Input (grams)"].notnull()
        & insulinDf["BWZ Carb Input (grams)"]
        != 0
    ]
    insulinMealDates = pandas.DataFrame(insulinneutralize["Date_Time"])

    insulinMealDates["DiffwBelow"] = (
        insulinMealDates.iloc[:, 0].diff(-1).dt.total_seconds()/3600
    )
    insulinMealDates = insulinMealDates.loc[insulinMealDates["DiffwBelow"] <= -2]
    insulinNoMealDates = insulinMealDates.loc[insulinMealDates["DiffwBelow"] <= -4]

    mealDatesList = cleaning(insulinMealDates, cgmDf)
    for mealDateTime in mealDatesList:
        idx = cgmDf[cgmDf["Date_Time"] == mealDateTime]["Sensor Glucose (mg/dL)"].index[
            0
        ]
        mealDataMatrix.append(
            list(
                cgmDf["Sensor Glucose (mg/dL)"]
                .iloc[cgmDf.shape[0] - 1 - idx - 6 : cgmDf.shape[0] - 1 - idx + 24]
                .values
            )
        )
   

    noMealDatesList = cleaning(insulinNoMealDates, cgmDf)
    for noMealDateTime in noMealDatesList:
        idx = cgmDf[cgmDf["Date_Time"] == noMealDateTime][
            "Sensor Glucose (mg/dL)"
        ].index[0]
        noMealDataMatrix.append(
            list(
                cgmDf["Sensor Glucose (mg/dL)"]
                .iloc[cgmDf.shape[0] - 1 - idx + 24 : cgmDf.shape[0] - 1 - idx + 48]
                .values
            )
        )

    mealDf = pandas.DataFrame(mealDataMatrix).dropna()
    noMealDf = pandas.DataFrame(noMealDataMatrix).dropna()
    return (mealDf, noMealDf)

def cleaning(arg0, cgmDf):
    arg0.drop(arg0.head(1).index, inplace=True)
    arg0.drop(arg0.tail(2).index, inplace=True)
    return [
        cgmDf.loc[cgmDf['Date_Time'] >= arg0['Date_Time'][ind]][
            'Date_Time'
        ].iloc[0]
        for ind in arg0.index
    ]


# In[88]:


mealDf,noMealDf=makeDf(insulinDf,cgmDf)
mealDfset2, noMealDfset2=makeDf(insulinDfset2,cgmDfset2)

meal = pandas.concat([mealDf, mealDfset2], ignore_index=True, sort = False)
no_meal = pandas.concat([noMealDf, noMealDfset2], ignore_index=True, sort = False)


# In[89]:


def createmealfeaturematrix(data):
    op_matrix = pd.DataFrame()
    # time between max and min glucose
    op_matrix["Time_bet_max_min"] = (
        data.iloc[:, 22:25].idxmin(axis=1) - data.iloc[:, 5:19].idxmax(axis=1)
    ) * 5
    
    # normalize glucose difference
    op_matrix["Glucose_Difference_normalized"] = (
        data.iloc[:, 5:19].max(axis=1) - data.iloc[:, 22:25].min(axis=1)
    ) / (data.iloc[:, 22:25].min(axis=1))

    # windowed mean
    if data.shape[1] > 24:
        for i in range(6, data.shape[1], 6):
            op_matrix["Mean_" + str(i - 6)] = data.iloc[
                :, i : i + 6
            ].mean(axis=1)
    else:
        for i in range(0, data.shape[1], 6):
            op_matrix["Mean_" + str(i)] = data.iloc[:, i : i + 6].mean(
                axis=1
            )
            
    #entropy
    """ def get_entropy(series):
        series_counts = series.value_counts()
        return scipy.stats.entropy(series_counts)
    
    op_matrix['Entropy'] = data.apply(lambda row: get_entropy(row), axis=1) """
    

    # cgm velocity
    velocityDF = pd.DataFrame()
    for i in range(0, data.shape[1] - 5):
        velocityDF["Vel_" + str(i)] = data.iloc[:, i + 5] - data.iloc[:, i]
    op_matrix["Window_Velocity_Max"] = velocityDF.max(axis=1, skipna=True)
    
    #1st and 2nd differential
    tm=data.iloc[:,22:25].idxmin(axis=1)
    maximum=data.iloc[:,5:19].idxmax(axis=1)
    list1=[]
    second_differential_data=[]
    standard_deviation=[]
    for i in range(len(data)):
        list1.append(np.diff(data.iloc[:,maximum[i]:tm[i]].iloc[i].tolist()).max())
        second_differential_data.append(np.diff(np.diff(data.iloc[:,maximum[i]:tm[i]].iloc[i].tolist())).max())
        standard_deviation.append(np.std(data.iloc[i]))
    op_matrix['1stDifferential']=list1
    op_matrix['2ndDifferential']=second_differential_data

    # fft_4
    def get_fft(row):
        cgmFFTValues = np.abs(fft(row))
        cgmFFTValues.sort()
        return np.flip(cgmFFTValues)[0:4]
    
    FFT = pd.DataFrame()
    FFT["FFT_Top2"] = data.apply(lambda row: get_fft(row), axis=1)
    FFT_updated = pd.DataFrame(
        FFT.FFT_Top2.tolist(), columns=["FFT_2", "FFT_3", "FFT_4","FFT_5"]
    )
    FFT_updated.head()
    for i in range(1,5):
        op_matrix["FFT_" + str(i + 1)] = FFT_updated["FFT_" + str(i + 1)]

    return op_matrix


# In[90]:


Meal_features=createmealfeaturematrix(meal)
Meal_features


# In[91]:


No_Meal_features=createmealfeaturematrix(no_meal)
No_Meal_features


# In[92]:


pca = PCA(n_components=5)
principalComponents = pca.fit(Meal_features)
PCA_mealdata = pca.fit_transform(Meal_features)

principalComponents = pca.fit(No_Meal_features)
PCA_nomealdata = pca.fit_transform(No_Meal_features)


# In[93]:


Training_data = np.concatenate((PCA_mealdata, PCA_nomealdata), axis=0)
no_of_mealrows = meal.shape[0]
no_of_nomealrows = no_meal.shape[0]
Training_labels = [1 for _ in range(no_of_mealrows)]
for _ in range(no_of_nomealrows):
    Training_labels.append(0)


# In[95]:


A=pd.DataFrame(Training_data)
B=pd.DataFrame(Training_labels)
FM = pd.concat([A,B], axis=1, sort=False)

Total_Data = FM.reindex(np.random.permutation(FM.index))

kf = RepeatedKFold(n_splits=5, n_repeats=5)

for train_index, test_index in kf.split(Total_Data):
    trainData = Total_Data.iloc[train_index, 0:5]
    trainLabel = Total_Data.iloc[train_index, 5]
    testData = Total_Data.iloc[test_index, 0:5]
    testLabel = Total_Data.iloc[test_index, 5]

    clf = tree.DecisionTreeClassifier()
    clf.fit(trainData, trainLabel)
    #predictedLabel = clf.predict(testData)
    
    pickle.dump(clf, open("model.pkl", 'wb'))

