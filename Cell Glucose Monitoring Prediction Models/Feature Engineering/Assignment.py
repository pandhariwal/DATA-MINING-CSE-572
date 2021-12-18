from copy import deepcopy
import datetime

import pandas as pd
from types import DynamicClassAttribute
import csv

file=''
file1="E:\photocopy\Vinay Pandhariwal\COLLEGE\ASU\Courses\DM\Assignment1\\" 


####################################################
## To change the date in a common format.
###################################################

def normalizedate(date,time):
    hr,minu,sec=time.split(':')
    try:
        month, day, year = date.split('/')
        abc=datetime.datetime(int(year), int(month), int(day),int(hr),int(minu),int(sec))
    except ValueError:
        month, day, year =date.split('-')
        abc=datetime.datetime(int(year), int(month), int(day),int(hr),int(minu),int(sec))
 
    return abc

###################################################
## To find out the auto trigger date from th insulin.csv file.
###################################################

def autodatemethod():
    with open(file + 'InsulinData.csv', newline='') as csvfile1:
        reader1=pd.read_csv(csvfile1)
        autodate=''
        for index,row1 in reader1.iterrows():
            if(row1['Alarm']=='AUTO MODE ACTIVE PLGM OFF'):
                if(autodate!=''):
                    if autodate>normalizedate(row1['Date'], row1['Time']):
                        autodate=normalizedate(row1['Date'], row1['Time'])
                else:
                    autodate=normalizedate(row1['Date'],row1['Time'])
    return autodate

######################################################
##Function to check the date change
######################################################

def datechange(od,nd):
    if(nd.date()==od.date()):
        return False
    else:
        return True


######################################################
## Main Driver Code
######################################################

def processvals():
    with open(file +'CGMData.csv', newline='') as csvfile:
        reader = pd.read_csv(csvfile,keep_default_na=False,index_col=False)
        autodate=autodatemethod()
        od,ld,fd=('' for i in range(3))
        flag=1
        
        OR,DR,WR,PROA,PRDA,PRWA,PROM,PRDM,PRWM=[[[] for i in range(6)] for j in range(9)]
        for index,row in reader.iterrows():
            nd=normalizedate(row['Date'],row['Time'])
            if(od==''):
                od=nd
            if(ld==''):
                ld=nd
  
            if(nd>autodate):                
                if(not datechange(od,nd)):
                    val= row['Sensor Glucose (mg/dL)']
                    if(not(val=='')):
                        val=int(val)
                        OR,DR,WR=putval(OR,DR,WR,val,nd)
                else:
                    PROA,PRDA,PRWA=append(PROA,PRDA,PRWA,OR,DR,WR)
                    OR.clear()
                    DR.clear()
                    WR.clear()
                    OR,DR,WR=[[[] for i in range(6)] for j in range(3)]
                    od=nd
                    if(not(val=='')):
                        OR,DR,WR=putval(OR,DR,WR,val,nd)
            elif(flag==1 and nd<autodate):
                flag=0
                OR.clear()
                DR.clear()
                WR.clear()
                OR,DR,WR=[[[] for i in range(6)] for j in range(3)]
                od=nd
            
            if(nd<=autodate):
                if(datechange(od,nd)==False):
                    val= row['Sensor Glucose (mg/dL)']
                    if(not(val=='')):
                        val=int(val)
                        OR,DR,WR=putval(OR,DR,WR,val,nd)
                else:
                    PROM,PRDM,PRWM=append(PROM,PRDM,PRWM,OR,DR,WR)
                    OR.clear()
                    DR.clear()
                    WR.clear()
                    OR,DR,WR=[[[] for i in range(6)] for j in range(3)]
                    od=nd
                    if(not(val=='')):
                        OR,DR,WR=putval(OR,DR,WR,val,nd)
            
        fd=nd
        writeresult(PROA,PRDA,PRWA,PROM,PRDM,PRWM,fd,ld)            
                    
         
######################################################
##Function to write to result.csv
######################################################

def writeresult(PROA,PRDA,PRWA,PROM,PRDM,PRWM,fd,ld):
        y=ld.date()-autodatemethod().date()
        x=y.days+1
        z=autodatemethod().date()-fd.date()
        a=z.days+1
        
        #rows = [['Manual Mode',sum(PROM[0])/a,sum(PROM[1])/a,sum(PROM[2])/a,sum(PROM[3])/a,sum(PROM[4])/a,sum(PROM[5])/a,sum(PRDM[0])/a,sum(PRDM[1])/a,sum(PRDM[2])/a,sum(PRDM[3])/a,sum(PRDM[4])/a,sum(PRDM[5])/a,sum(PRWM[0])/a,sum(PRWM[1])/a,sum(PRWM[2])/a,sum(PRWM[3])/a,sum(PRWM[4])/a,sum(PRWM[5])/a],
        #        ['Auto Mode',sum(PROA[0])/x,sum(PROA[1])/x,sum(PROA[2])/x,sum(PROA[3])/x,sum(PROA[4])/x,sum(PROA[5])/x,sum(PRDA[0])/x,sum(PRDA[1])/x,sum(PRDA[2])/x,sum(PRDA[3])/x,sum(PRDA[4])/x,sum(PRDA[5])/x,sum(PRWA[0])/x,sum(PRWA[1])/x,sum(PRWA[2])/x,sum(PRWA[3])/x,sum(PRWA[4])/x,sum(PRWA[5])/x]]
        #fields = ['','Overnight Percentage time in hyperglycemia (CGM > 180 mg/dL)','Overnight percentage of time in hyperglycemia critical (CGM > 250 mg/dL)','Overnight percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)','Overnight percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)','Overnight percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)','Overnight percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)','Daytime Percentage time in hyperglycemia (CGM > 180 mg/dL)','Daytime percentage of time in hyperglycemia critical (CGM > 250 mg/dL)','Daytime percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)','Daytime percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)','Daytime percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)','Daytime percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)','Whole Day Percentage time in hyperglycemia (CGM > 180 mg/dL)','Whole day percentage of time in hyperglycemia critical (CGM > 250 mg/dL)','Whole day percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)','Whole day percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)','Whole day percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)','Whole Day percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)'] 
        
        rows = [[sum(PROM[0])/a,sum(PROM[1])/a,sum(PROM[2])/a,sum(PROM[3])/a,sum(PROM[4])/a,sum(PROM[5])/a,sum(PRDM[0])/a,sum(PRDM[1])/a,sum(PRDM[2])/a,sum(PRDM[3])/a,sum(PRDM[4])/a,sum(PRDM[5])/a,sum(PRWM[0])/a,sum(PRWM[1])/a,sum(PRWM[2])/a,sum(PRWM[3])/a,sum(PRWM[4])/a,sum(PRWM[5])/a,1.1],
                [sum(PROA[0])/x,sum(PROA[1])/x,sum(PROA[2])/x,sum(PROA[3])/x,sum(PROA[4])/x,sum(PROA[5])/x,sum(PRDA[0])/x,sum(PRDA[1])/x,sum(PRDA[2])/x,sum(PRDA[3])/x,sum(PRDA[4])/x,sum(PRDA[5])/x,sum(PRWA[0])/x,sum(PRWA[1])/x,sum(PRWA[2])/x,sum(PRWA[3])/x,sum(PRWA[4])/x,sum(PRWA[5])/x,1.1]]

        with open(file +"Results.csv", 'w',newline='') as csvfile: 
            csvwriter = csv.writer(csvfile) 
            #csvwriter.writerow(fields)         
            csvwriter.writerows(rows)
        csvfile.close()


######################################################
##Function to append values
######################################################

def append(PROX,PRDX,PRWX,OR,DR,WR):
    PROX[0].append((len(OR[0])/288)*100)
    PROX[1].append((len(OR[1])/288)*100)
    PROX[2].append((len(OR[2])/288)*100)
    PROX[3].append((len(OR[3])/288)*100)
    PROX[4].append((len(OR[4])/288)*100)
    PROX[5].append((len(OR[5])/288)*100)
    
    PRDX[0].append((len(DR[0])/288)*100)
    PRDX[1].append((len(DR[1])/288)*100)
    PRDX[2].append((len(DR[2])/288)*100)
    PRDX[3].append((len(DR[3])/288)*100)
    PRDX[4].append((len(DR[4])/288)*100)
    PRDX[5].append((len(DR[5])/288)*100)
    
    PRWX[0].append((len(WR[0])/288)*100)
    PRWX[1].append((len(WR[1])/288)*100)
    PRWX[2].append((len(WR[2])/288)*100)
    PRWX[3].append((len(WR[3])/288)*100)
    PRWX[4].append((len(WR[4])/288)*100)
    PRWX[5].append((len(WR[5])/288)*100)
    return PROX,PRDX,PRWX
    

######################################################
##Function to check day or night
######################################################

def chkovernight(stamp):
    if(stamp.time()<datetime.time(6,0,0) or datetime.time(23,59,0)<stamp.time()):
        return True
    else:
        return False    


######################################################
##Function to compare values
######################################################

def putval(OR,DR,WR,val,nd):
    
    if(chkovernight(nd)):
        if(val>180):
            OR[0].append(val)
            WR[0].append(val) 
        if(val>250):
            OR[1].append(val)
            WR[1].append(val)
        if(val>=70 and val<=180):
            OR[2].append(val)
            WR[2].append(val)
        if(val>=70 and val<=150):
            OR[3].append(val)
            WR[3].append(val)
        if(val<70 ):
            OR[4].append(val)
            WR[4].append(val)
        if(val<54):
            OR[5].append(val)
            WR[5].append(val)
    else:
        if(val>180):
            DR[0].append(val)
            WR[0].append(val) 
        if(val>250):
            DR[1].append(val)
            WR[1].append(val)
        if(val>=70 and val<=180):
            DR[2].append(val)
            WR[2].append(val)
        if(val>=70 and val<=150):
            DR[3].append(val)
            WR[3].append(val)
        if(val<70 ):
            DR[4].append(val)
            WR[4].append(val)
        if(val<54):
            DR[5].append(val)
            WR[5].append(val)
    return OR,DR,WR



processvals()
