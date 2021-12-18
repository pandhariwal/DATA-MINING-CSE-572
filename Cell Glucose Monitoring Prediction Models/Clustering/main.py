import pandas
import numpy as np
from numpy.fft import fft
from sklearn.decomposition import PCA
import math
from sklearn.cluster import KMeans,DBSCAN
binLen=0
minV=0
maxV=0

insulinDf = pandas.read_csv('InsulinData.csv',parse_dates=[['Date','Time']],low_memory=False,usecols=['Date','Time','BWZ Carb Input (grams)']).iloc[::-1]
cgmDf = pandas.read_csv('CGMData.csv',parse_dates=[['Date','Time']],low_memory=False,usecols=['Date','Time','Sensor Glucose (mg/dL)']).iloc[::-1]

def makeDf(insulinDF, cgmDf):
    global minV,maxV,binLen
    mealDataMatrix = []
    insulinneutralize = insulinDf[
        insulinDf["BWZ Carb Input (grams)"].notnull()
        & insulinDf["BWZ Carb Input (grams)"]
        != 0
    ]
    insulinMealDates = pandas.DataFrame(insulinneutralize[["Date_Time",'BWZ Carb Input (grams)']])

    insulinMealDates["DiffwBelow"] = (
        insulinMealDates.iloc[:, 0].diff(-1).dt.total_seconds()/3600
    )
    insulinMealDates = insulinMealDates.loc[insulinMealDates["DiffwBelow"] <= -2]
    mealDatesList = cleaning(insulinMealDates, cgmDf)

    for mealDateTime in mealDatesList:
        idx = cgmDf[cgmDf['Date_Time'] == mealDateTime[0]]['Sensor Glucose (mg/dL)'].index[0]
        l = list(cgmDf['Sensor Glucose (mg/dL)'].iloc[cgmDf.shape[0]-1-idx-6:cgmDf.shape[0]-1-idx+24].values)
        l.append(mealDateTime[1])
        mealDataMatrix.append(l)
    mealDf = pandas.DataFrame(mealDataMatrix).dropna()
    mealDf = mealDf.reset_index(drop=True)
    return (mealDf)

def cleaning(insulinMealDates, cgmDf):
    global binLen, minV, maxV, nBins
    insulinMealDates.drop(insulinMealDates.head(1).index, inplace=True)
    insulinMealDates.drop(insulinMealDates.tail(2).index, inplace=True)
    binLen = 20
    minV = insulinMealDates['BWZ Carb Input (grams)'].min()
    maxV = insulinMealDates['BWZ Carb Input (grams)'].max()

    nBins = (int)((maxV - minV)/20)

    for ind in insulinMealDates.index:
        insulinMealDates['BWZ Carb Input (grams)'][ind] = (int)(insulinMealDates['BWZ Carb Input (grams)'][ind]/(minV + 20))

    mealDatesList = []
    for ind in insulinMealDates.index: 
        l = [
            (
                cgmDf.loc[
                    cgmDf['Date_Time'] >= insulinMealDates['Date_Time'][ind]
                ]
            )['Date_Time'].iloc[0],
            insulinMealDates['BWZ Carb Input (grams)'][ind],
        ]

        mealDatesList.append(l)
    return mealDatesList

mealDf=makeDf(insulinDf,cgmDf)

def createmealfeaturematrix(data):
    op_matrix = pandas.DataFrame()
    
    op_matrix["Time_bet_max_min"] = (
        data.iloc[:, 22:25].idxmin(axis=1) - data.iloc[:, 5:19].idxmax(axis=1)
    ) * 5
    
    op_matrix["Glucose_Difference_normalized"] = (
        data.iloc[:, 5:19].max(axis=1) - data.iloc[:, 22:25].min(axis=1)
    ) / (data.iloc[:, 22:25].min(axis=1))

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
            
    velocityDF = pandas.DataFrame()
    for i in range(0, data.shape[1] - 5):
        velocityDF["Vel_" + str(i)] = data.iloc[:, i + 5] - data.iloc[:, i]
    op_matrix["Window_Velocity_Max"] = velocityDF.max(axis=1, skipna=True)
    
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

    def get_fft(row):
        cgmFFTValues = np.abs(fft(row))
        cgmFFTValues.sort()
        return np.flip(cgmFFTValues)[0:4]
    
    FFT = pandas.DataFrame()
    FFT["FFT_Top2"] = data.apply(lambda row: get_fft(row), axis=1)
    FFT_updated = pandas.DataFrame(
        FFT.FFT_Top2.tolist(), columns=["FFT_2", "FFT_3", "FFT_4","FFT_5"]
    )
    FFT_updated.head()
    for i in range(1,5):
        op_matrix["FFT_" + str(i + 1)] = FFT_updated["FFT_" + str(i + 1)]

    return op_matrix

Meal_features=createmealfeaturematrix(mealDf)

pca = PCA(n_components=10)
principalComponents = pca.fit(Meal_features)
PCA_mealdata = pca.fit_transform(Meal_features)

kmeans=KMeans(n_clusters=nBins, random_state=0).fit(PCA_mealdata)
SSE_KMeans=kmeans.inertia_

clusterBinMatrix = [[0 for i in range(nBins)] for i in range(nBins)]
for i in range(kmeans.labels_.shape[0]):
    clusterBinMatrix[kmeans.labels_[i]][int(mealDf.iloc[i][30])] += 1
Kmeans_Entropy =[0,0,0,0,0,0]
Kmeans_Purity = 0
totalPoints = sum(sum(clusterBinMatrix,[]))
for i in range(len(clusterBinMatrix)):
    Kmeans_Purity += (max(clusterBinMatrix[i])*1.0)/(totalPoints*1.0)
    for j in range(len(clusterBinMatrix[i])):
        P = (clusterBinMatrix[i][j]*1.0)/(sum(clusterBinMatrix[i])*1.0)
        if(P != 0):
            Kmeans_Entropy[i] += (-P) * math.log(P,2) * (sum(clusterBinMatrix[i])*1.0/(totalPoints)*1.0)    
Kmeans_Entropy = sum(Kmeans_Entropy)

dbscan = DBSCAN(eps = 210, min_samples = 6).fit(PCA_mealdata) 

for i in range(dbscan.labels_.size):
    if dbscan.labels_[i] == -1:
        min = float('inf')
        l = -1
        for j in range(dbscan.labels_.size):
            if dbscan.labels_[j] != -1:
                eucDist = np.linalg.norm(PCA_mealdata[i] - PCA_mealdata[j])
                if eucDist < min:
                    min = eucDist
                    l = dbscan.labels_[j]
        dbscan.labels_[i] = l            
        
dbscan.labels_ = np.array(dbscan.labels_)

dbScanSSE = 0
for i in range(nBins):
    cluster = PCA_mealdata[dbscan.labels_ == i]
    clusterMean = cluster.mean(axis = 0)
    dbScanSSE += ((cluster - clusterMean) ** 2).sum()

clusterBinMatrix = [[0 for i in range(nBins)] for i in range(nBins)]
for i in range(dbscan.labels_.shape[0]):
    clusterBinMatrix[dbscan.labels_[i]][int(mealDf.iloc[i][30])] += 1
DbScanEntropy = [0,0,0,0,0,0]
DbScanPurity = 0
totalPoints = sum(sum(clusterBinMatrix,[]))
for i in range(len(clusterBinMatrix)):
    DbScanPurity += (max(clusterBinMatrix[i])*1.0)/(totalPoints*1.0)
    for j in range(len(clusterBinMatrix[i])):
        P = (clusterBinMatrix[i][j]*1.0)/(sum(clusterBinMatrix[i])*1.0)
        if(P != 0):
            DbScanEntropy[i] += (-P) * math.log(P,2) * (sum(clusterBinMatrix[i])*1.0/(totalPoints)*1.0)    
DbScanEntropy = sum(DbScanEntropy)

result = [
    [SSE_KMeans,
    dbScanSSE,
    Kmeans_Entropy,
    DbScanEntropy,
    Kmeans_Purity,
    DbScanPurity]
]

resultDf = pandas.DataFrame(result)
resultDf.to_csv('Results.csv',float_format='%.6f',index=False, header=False)

