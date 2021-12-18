import pandas as pd
import numpy as np
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
#from collections import Counter
import math
import warnings
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from statistics import mode

warnings.filterwarnings("ignore")
    
def getMealtimes(new_insulin_data):
    mytimes =[]
    insulinValues =[]
    insulinLevels =[]
    newTimes1 =[]
    newTimes2 =[]
    mealTimes = []
    diff =[]
    column_mine = new_insulin_data['BWZ Carb Input (grams)']
    maxValue= column_mine.max()
    #print (maxValue)
    minValue = column_mine.min()
    #print (minValue)
    bins = math.ceil(maxValue-minValue/60)
    #print (bins)    
    for i in new_insulin_data['datetime']:
        mytimes.append(i)
    for i in new_insulin_data['BWZ Carb Input (grams)']:
        insulinValues.append(i)
    for i,j in enumerate(mytimes):
        if(i<len(mytimes)-1):
            diff.append((mytimes[i+1]-mytimes[i]).total_seconds()/3600)
    newTimes1 = mytimes[0:-1]
    newTimes2 = mytimes[1:]
    bins=[]
    for i in insulinValues[0:-1]:
        bins.append(0 if (i>=minValue and i<=minValue+20) else 1 if (i>=minValue+21 and i<=minValue+40) else 2 if(i>=minValue+41 and i<=minValue+60) else 3 if(i>=minValue+61 and i<=minValue+80) else 4 if(i>=minValue+81 and i<=minValue+100) else 5 if(i>=minValue+101 and i<=minValue+120) else 6)
    reqValues = list(zip(newTimes1, newTimes2, diff,bins))
    for j in reqValues:
        if j[2]>2.5:
            mealTimes.append(j[0])
            insulinLevels.append(j[3])
        else:
            continue
    return mealTimes,insulinLevels

def getMealData(mealTimes,startTime,endTime,insulinLevels,new_glucose_data):
    newMealDataRows = []
    for j,newTime in enumerate(mealTimes):
        meal_index_start= new_glucose_data[new_glucose_data['datetime'].between(newTime+ pd.DateOffset(hours=startTime),newTime + pd.DateOffset(hours=endTime))]
        
        if meal_index_start.shape[0]<8:
            del insulinLevels[j]
            continue
        glucoseValues = meal_index_start['Sensor Glucose (mg/dL)'].to_numpy()
        mean = meal_index_start['Sensor Glucose (mg/dL)'].mean()
        missing_values_count = 30 - len(glucoseValues)
        if missing_values_count > 0:
            for i in range(missing_values_count):
                glucoseValues = np.append(glucoseValues, mean)
        newMealDataRows.append(glucoseValues[0:30])
    return pd.DataFrame(data=newMealDataRows),insulinLevels
        
       
def processData(insulin_data,glucose_data):
    mealData = pd.DataFrame()
    glucose_data['Sensor Glucose (mg/dL)'] = glucose_data['Sensor Glucose (mg/dL)'].interpolate(method='linear',limit_direction = 'both')
    insulin_data= insulin_data[::-1]
    glucose_data= glucose_data[::-1]
    insulin_data['datetime']= insulin_data['Date']+" "+insulin_data['Time']
    insulin_data['datetime']=pd.to_datetime(insulin_data['datetime'])
    glucose_data['datetime']= glucose_data['Date']+" "+glucose_data['Time']
    glucose_data['datetime']=pd.to_datetime(insulin_data['datetime'])
    
    new_insulin_data = insulin_data[['datetime','BWZ Carb Input (grams)']]
    new_glucose_data = glucose_data[['datetime','Sensor Glucose (mg/dL)']]

    new_insulin_data1 = new_insulin_data[(new_insulin_data['BWZ Carb Input (grams)']>0) ]
    mealTimes,insulinLevels = getMealtimes(new_insulin_data1)
    mealData,new_insulinLevels = getMealData(mealTimes,-0.5,2,insulinLevels,new_glucose_data)
    #concat_df=pd.concat([mealData,pd.DataFrame(new_insulinLevels)],axis=1,sort=False,ignore_index=True)
    #concat_df.to_csv(r'combined_df.csv', index = False)
    #print(Counter(list(concat_df[30])))

    return mealData,new_insulinLevels
def absoluteValueMean(param):
    meanValue = 0
    for p in range(0, len(param) - 1):
        meanValue = meanValue + np.abs(param[(p + 1)] - param[p])
    return meanValue / len(param)

def glucoseEntropy(param):
    paramLen = len(param)
    entropy = 0
    if paramLen <= 1:
        return 0
    else:
        value, count = np.unique(param, return_counts=True)
        ratio = count / paramLen
        nonZero_ratio = np.count_nonzero(ratio)
        if nonZero_ratio <= 1:
            return 0
        for i in ratio:
            entropy -= i * np.log2(i)
        return entropy

def rootMeanSquare(param):
    rootMeanSquare = 0
    for p in range(0, len(param) - 1):
        
        rootMeanSquare = rootMeanSquare + np.square(param[p])
    return np.sqrt(rootMeanSquare / len(param))


def fastFourier(param):
    fastFourier = fft(param)
    paramLen = len(param)
    t = 2/300
    amplitude = []
    frequency = np.linspace(0, paramLen * t, paramLen)
    for amp in fastFourier:
        amplitude.append(np.abs(amp))
    sortedAmplitude = amplitude
    sortedAmplitude = sorted(sortedAmplitude)
    max_amplitude = sortedAmplitude[(-2)]
    max_frequency = frequency.tolist()[amplitude.index(max_amplitude)]
    return [max_amplitude, max_frequency]




def glucoseFeatures(meal_Nomeal_data):
    glucoseFeatures=pd.DataFrame()
    for i in range(0, meal_Nomeal_data.shape[0]):
        param = meal_Nomeal_data.iloc[i, :].tolist()
        glucoseFeatures = glucoseFeatures.append({ 
         'Minimum Value':min(param), 
         'Maximum Value':max(param),
         'Mean of Absolute Values1':absoluteValueMean(param[:13]), 
         'Mean of Absolute Values2':absoluteValueMean(param[13:]), 
         'Max_Zero_Crossing':fn_zero_crossings(param, meal_Nomeal_data.shape[1])[0], 
         'Zero_Crossing_Rate':fn_zero_crossings(param, meal_Nomeal_data.shape[1])[1], 
         'Root Mean Square':rootMeanSquare(param),
         'Entropy':rootMeanSquare(param), 
         'Max FFT Amplitude1':fastFourier(param[:13])[0], 
         'Max FFT Frequency1':fastFourier(param[:13])[1], 
         'Max FFT Amplitude2':fastFourier(param[13:])[0], 
         'Max FFT Frequency2':fastFourier(param[13:])[1]},
          ignore_index=True)
    return glucoseFeatures
def fn_zero_crossings(row, xAxis):
    slopes = [
     0]
    zero_cross = list()
    zero_crossing_rate = 0
    X = [i for i in range(xAxis)][::-1]
    Y = row[::-1]
    for index in range(0, len(X) - 1):
        slopes.append((Y[(index + 1)] - Y[index]) / (X[(index + 1)] - X[index]))

    for index in range(0, len(slopes) - 1):
        if slopes[index] * slopes[(index + 1)] < 0:
            zero_cross.append([slopes[(index + 1)] - slopes[index], X[(index + 1)]])

    zero_crossing_rate = np.sum([np.abs(np.sign(slopes[(i + 1)]) - np.sign(slopes[i])) for i in range(0, len(slopes) - 1)]) / (2 * len(slopes))
    if len(zero_cross) > 0:
        return [max(zero_cross)[0], zero_crossing_rate]
    else:
        return [
         0, 0]
def getFeatures(mealData):
    mealDataFeatures = glucoseFeatures(mealData.iloc[:,:-1])
    
    
    stdScaler = StandardScaler()
    meal_std = stdScaler.fit_transform(mealDataFeatures)
    
    pca = PCA(n_components=12)
    pca.fit(meal_std)
    
    with open('pcs_glucose_data.pkl', 'wb') as (file):
        pickle.dump(pca, file)
        
    meal_pca = pd.DataFrame(pca.fit_transform(meal_std))
    return meal_pca
def compute_Entropy(bins):
    mealEntropy= []
    for insulinBin in bins:
    	insulinBin = np.array(insulinBin)
    	insulinBin = insulinBin / float(insulinBin.sum())
    	binEntropy = (insulinBin * [ np.log2(glucose) if glucose!=0 else 0 for glucose in insulinBin]).sum()
    	mealEntropy += [binEntropy]
   
    return mealEntropy

def compute_Purity(bins):
    mealPurity = []
    for insulinBin in bins:
    	insulinBin = np.array(insulinBin)
    	insulinBin = insulinBin / float(insulinBin.sum())
    	binPurity = insulinBin.max()
    	mealPurity += [binPurity]
    return mealPurity
def computeDBSCAN_SSE(dbscan_sse,test,meal_pca2):
        for i in test.index:
            dbscan_sse=0
            for index,row in meal_pca2[meal_pca2['clusters']==i].iterrows(): 
                test_row=list(test.iloc[0,:])
                meal_row=list(row[:-1])
                for j in range(0,12):
                    dbscan_sse+=((test_row[j]-meal_row[j])**2)
        return dbscan_sse
def clusterMatrixwithGroundTruth(groundTruth,Clustered,k):
    clusterMatrix= np.zeros((k, k))
    for i,j in enumerate(groundTruth):
         val1 = j
         val2 = Clustered[i]
         clusterMatrix[val1,val2]+=1
    return clusterMatrix
#Grid search for getting better epsilon values
#def dbscan_grid_search(X_data, lst, clst_count, eps_space = 0.5, min_samples_space = 5, min_clust = 0, max_clust = 10):
#    n_iterations = 0
#    for eps_val in eps_space:
#        for samples_val in min_samples_space:
#            dbscan_grid = DBSCAN(eps = eps_val, min_samples = samples_val,algorithm='kd_tree')
#            clusters = dbscan_grid.fit_predict(X = X_data)
#            cluster_count = Counter(clusters).most_common()
#            n_clusters = sum(abs(pd.np.unique(clusters))) - 1
#            n_iterations += 1
#            if n_clusters >= min_clust and n_clusters <= max_clust:
#                clusters.append([eps_val, samples_val, n_clusters])
#                clst_count.append(cluster_count)
#    print(f"""Search Complete. \nYour list is now of length {len(lst)}. """)
#    print(f"""Hyperparameter combinations checked: {n_iterations}. \n""")
#    return clusters,clst_count
        
if __name__=='__main__':
    
    #reading Patient 1 data and getting the meal features
    insulin_data=pd.read_csv("InsulinData.csv",low_memory=False)
    glucose_data=pd.read_csv("CGMData.csv",low_memory=False)
    patient_data,insulinLevels = processData(insulin_data,glucose_data)
    meal_pca = getFeatures(patient_data)
    #    print(help(KMeans))
    
    #Performing K-means
    kmeans = KMeans(n_clusters=7,max_iter=7000)
    kmeans.fit_predict(meal_pca)
    pLabels=list(kmeans.labels_)
    df = pd.DataFrame()
    df['bins']=insulinLevels
    df['kmeans_clusters']=pLabels 
    #print(Counter(list(df['kmeans_clusters'])))
    clusterMatrix = clusterMatrixwithGroundTruth(df['bins'],df['kmeans_clusters'],7)
    cluster_entropy = compute_Entropy(clusterMatrix)
    cluster_purity = compute_Purity(clusterMatrix)
    totalCount = np.array([insulinBin.sum() for insulinBin in clusterMatrix])
    binCount = totalCount / float(totalCount.sum())
    
    
    #Kmeans
    kmeans_SSE = kmeans.inertia_
    kmeans_purity =  (cluster_purity*binCount).sum()
    kmeans_entropy = -(cluster_entropy*binCount).sum()
    
    
    
  
    #DBSCAN Implementation
    dbscan_df=pd.DataFrame()
    db = DBSCAN(eps=2.8000000000000003,min_samples=2,algorithm='kd_tree')
    clusters = db.fit_predict(meal_pca)
    #dbscan_labels= db.labels_
    dbscan_df=pd.DataFrame({'pc1':list(meal_pca.iloc[:,0]),'pc2':list(meal_pca.iloc[:,1]),'clusters':list(clusters)})
    outliers_df=dbscan_df[dbscan_df['clusters']==-1].iloc[:,0:2]
    #print(Counter(list(dbscan_df['clusters'])))
    
    
    #Assigning outliers to neighrest neighbors using KNN
    knn = KNeighborsClassifier(n_neighbors=4,p=2)
    knn.fit(dbscan_df[dbscan_df['clusters']!=-1].iloc[:,0:2],dbscan_df[dbscan_df['clusters']!=-1].iloc[:,2])
    for x,y in zip(outliers_df.iloc[:,0],outliers_df.iloc[:,1]):
        dbscan_df.loc[(dbscan_df['pc1'] == x) & (dbscan_df['pc2'] == y),'clusters']=knn.predict([[x,y]])[0]
    
    #Applying bisecting k-means on the obtained clusters until number of bins is equal to ground truth
    initial_value=0
    bins = 7
    i = max(dbscan_df['clusters'])
    while i<bins-1:
        largestClusterLabel=mode(dbscan_df['clusters'])
        biCluster_df=dbscan_df[dbscan_df['clusters']==mode(dbscan_df['clusters'])]
        bi_kmeans = KMeans(n_clusters=2,max_iter=1000, algorithm = 'auto').fit(biCluster_df)
        #bi_centroids = bi_kmeans.cluster_centers_
        bi_pLabels=list(bi_kmeans.labels_)
        biCluster_df['bi_pcluster']=bi_pLabels
        biCluster_df=biCluster_df.replace(to_replace =0,  value =largestClusterLabel) 
        biCluster_df=biCluster_df.replace(to_replace =1,  value =max(dbscan_df['clusters'])+1) 
        #print (max(dbscan_df['clusters']))
        for x,y in zip(biCluster_df['pc1'],biCluster_df['pc2']):
           newLabel=biCluster_df.loc[(biCluster_df['pc1'] == x) & (biCluster_df['pc2'] == y)]
           dbscan_df.loc[(dbscan_df['pc1'] == x) & (dbscan_df['pc2'] == y),'clusters']=newLabel['bi_pcluster']
        df['clusters']=dbscan_df['clusters']
        i+=1

    #creating a matrix using groundtruth and dbscan clusters
    clusterMatrix_dbscan = clusterMatrixwithGroundTruth(df['bins'],dbscan_df['clusters'],7)
    
    cluster_entropy_db = compute_Entropy(clusterMatrix_dbscan)
    cluster_purity_db = compute_Purity(clusterMatrix_dbscan)
    totalCount = np.array([insulinBin.sum() for insulinBin in clusterMatrix_dbscan])
    binCount = totalCount / float(totalCount.sum())
    
    #print(Counter(list(dbscan_df['clusters'])))
    
    #combining meal_features with bins
    meal_pca2= meal_pca. join(dbscan_df['clusters']) 
    
    #getting centroid of each bin
    centroids = meal_pca2.groupby(dbscan_df['clusters']).mean()
    
    #DB Scan sse, entropy and purity
    dbscan_sse = computeDBSCAN_SSE(initial_value,centroids.iloc[:, : 12],meal_pca2)
    dbscan_purity =  (cluster_purity_db*binCount).sum()        
    dbscan_entropy = -(cluster_entropy_db*binCount).sum()
            
            

    #writing results to csv
    outputdf = pd.DataFrame([[kmeans_SSE,dbscan_sse,kmeans_entropy,dbscan_entropy,kmeans_purity,dbscan_purity]],columns=['K-Means SSE','DBSCAN SSE','K-Means entropy','DBSCAN entropy','K-Means purity','DBSCAN purity'])
    outputdf=outputdf.fillna(0)
    outputdf.to_csv('Results.csv', header=False, index=False)