#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import pandas
import numpy as np
import matplotlib.pyplot as plt
import pywt
from numpy.fft import fft
from sklearn.model_selection import train_test_split
import scipy
import scipy.fftpack
from sklearn.decomposition import PCA
import pickle
import pickle_compat

pickle_compat.patch()


# In[24]:


testSet = pd.read_csv("test.csv",header=None)


# In[25]:


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


# In[26]:


test_feature_matrix=pd.DataFrame()
test_feature_matrix=createmealfeaturematrix(testSet)


# In[27]:


test_feature_matrix


# In[30]:


pca = PCA(n_components=5)

principalComponents = pca.fit(test_feature_matrix)

testData = pca.fit_transform(test_feature_matrix)


# In[31]:


clf = pickle.load(open('model.pkl', 'rb'))
y_pred = clf.predict(testData)
new_y_pred = []
for i in y_pred:
    if(i > 0):
        new_y_pred.append(1)
    else:
        new_y_pred.append(0)
        
result = pd.DataFrame(new_y_pred)
result.to_csv('Results.csv', index=False)


# In[ ]:




