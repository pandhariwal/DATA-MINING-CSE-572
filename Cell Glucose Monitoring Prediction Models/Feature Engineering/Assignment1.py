from copy import deepcopy
import csv
import datetime
import pandas as pd


from types import DynamicClassAttribute

od=''
ld=''
fd=''
file='E:\photocopy\Vinay Pandhariwal\COLLEGE\ASU\Courses\DM\Assignment1'
autodate=''
List=[]

def normalizedate(date,time):
    hr,minu,sec=time.split(':')
    try:
        month, day, year = date.split('/')
        abc=datetime.datetime(int(year), int(month), int(day),int(hr),int(minu),int(sec))
    except ValueError:
        month, day, year =date.split('-')
        abc=datetime.datetime(int(year), int(month), int(day),int(hr),int(minu),int(sec))
 
    return abc

def autodatemethod():
    with open(file + '\InsulinData.csv', newline='') as csvfile1:
        reader1=pd.read_csv(csvfile1)
        global autodate
        for index,row1 in reader1.iterrows():
            if(row1['Alarm']=='AUTO MODE ACTIVE PLGM OFF'):
                if(autodate!=''):
                    if autodate>normalizedate(row1['Date'], row1['Time']):
                        autodate=normalizedate(row1['Date'], row1['Time'])
                else:
                    autodate=normalizedate(row1['Date'],row1['Time'])


def processvals():
    with open(file +'\CGMData.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        HP1,HPC1,R1,RS1,HOL11,HOL21,HP,HPC,R,RS,HOL1,HOL2 = invoke()
        HP1O,HPC1O,R1O,RS1O,HOL11O,HOL21O,HPO,HPCO,RO,RSO,HOL1O,HOL2O = invokeo()
        HP1D,HPC1D,R1D,RS1D,HOL11D,HOL21D,HPD,HPCD,RD,RSD,HOL1D,HOL2D = invoked()
        putval = newmethod394(RSD, HPD, HPC, HOL1O, RS, R, HOL1, HOL2, HPCO, HOL2O, RD, HOL1D, HP, RO, HOL2D, RSO, HPO, HPCD)
        
        reset = newmethod761(HPC, HOL2O, RD, R, HPCO, HOL2D, HPO, HP, RSO, HPCD, HPD, RSD, HOL1D, RS, HOL1, RO, HOL2, HOL1O)
        resetall= newmethod762(R1D, HP1D, HPC1D, HP1O, HOL21D, HP1, R1, HOL11O, HPC1, HOL11D, RS1, RS1D, R1O, HOL11, HOL21, RS1O, HPC1O, HOL21O)

        datechange = newmethod628()
        flag=1
        for row in reader:
            global ld,fd,od
            nd=normalizedate(row['Date'],row['Time'])
            if(nd>autodate):
                if(datechange(nd)==True):
                    append(HP1, HPC1, R1, RS1, HOL11, HOL21, HP, HPC, R, RS, HOL1, HOL2)
                    appendo(HP1O,HPC1O,R1O,RS1O,HOL11O,HOL21O,HPO,HPCO,RO,RSO,HOL1O,HOL2O)
                    appendd(HP1D,HPC1D,R1D,RS1D,HOL11D,HOL21D,HPD,HPCD,RD,RSD,HOL1D,HOL2D)
                    reset()
                    od=nd
                    val=row['Sensor Glucose (mg/dL)']
                    putval(val,nd)
                else:        
                    val=row['Sensor Glucose (mg/dL)']
                    putval(val,nd)
                fd=nd
            elif(not(nd>autodate) and flag==1):
                storeresultauto(R1D, HP1D, HPC1D, HP1O, HOL21D, HP1, R1, HOL11O, HPC1, HOL11D, RS1, RS1D, R1O, HOL11, HOL21, RS1O, HPC1O, HOL21O)
                reset()
                resetall()
                od=''
                fd=''
                flag=0

            if(nd<=autodate):
                if(datechange(nd)==True):
                    append(HP1, HPC1, R1, RS1, HOL11, HOL21, HP, HPC, R, RS, HOL1, HOL2)
                    appendo(HP1O,HPC1O,R1O,RS1O,HOL11O,HOL21O,HPO,HPCO,RO,RSO,HOL1O,HOL2O)
                    appendd(HP1D,HPC1D,R1D,RS1D,HOL11D,HOL21D,HPD,HPCD,RD,RSD,HOL1D,HOL2D)
                    reset()
                    od=nd
                    val=row['Sensor Glucose (mg/dL)']
                    putval(val,nd)
                else:        
                    val=row['Sensor Glucose (mg/dL)']
                    putval(val,nd)     
                fd=nd
    writeresultmanual(R1D, HP1D, HPC1D, HP1O, HOL21D, HP1, R1, HOL11O, HPC1, HOL11D, RS1, RS1D, R1O, HOL11, HOL21, RS1O, HPC1O, HOL21O)

def newmethod761(HPC, HOL2O, RD, R, HPCO, HOL2D, HPO, HP, RSO, HPCD, HPD, RSD, HOL1D, RS, HOL1, RO, HOL2, HOL1O):
    def reset():
        HP.clear()
        HPC.clear()
        R.clear()
        RS.clear()
        HOL1.clear()
        HOL2.clear()
        HPO.clear()
        HPCO.clear()
        RO.clear()
        RSO.clear()
        HOL1O.clear()
        HOL2O.clear()
        HPD.clear()
        HPCD.clear()
        RD.clear()
        RSD.clear()
        HOL1D.clear()
        HOL2D.clear()
    return reset

def newmethod762(R1D, HP1D, HPC1D, HP1O, HOL21D, HP1, R1, HOL11O, HPC1, HOL11D, RS1, RS1D, R1O, HOL11, HOL21, RS1O, HPC1O, HOL21O):
    def resetall():
        HP1.clear()
        HPC1.clear()
        R1.clear()
        RS1.clear()
        HOL11.clear()
        HOL21.clear()
        HP1O.clear()
        HPC1O.clear()
        R1O.clear()
        RS1O.clear()
        HOL11O.clear()
        HOL21O.clear()
        HP1D.clear()
        HPC1D.clear()
        R1D.clear()
        RS1D.clear()
        HOL11D.clear()
        HOL21D.clear()
    return resetall

def newmethod628():


def newmethod394(RSD, HPD, HPC, HOL1O, RS, R, HOL1, HOL2, HPCO, HOL2O, RD, HOL1D, HP, RO, HOL2D, RSO, HPO, HPCD):
    def putval(val,nd):
        if(val):
                val=int(val)
                if(chkovernight(nd)):
                    if(val>180):
                        HP.append(val)
                        HPO.append(val) 
                    if(val>250):
                        HPC.append(val)
                        HPCO.append(val)
                    if(val>=70 and val<=180):
                        R.append(val)
                        RO.append(val)
                    if(val>=70 and val<=150):
                        RS.append(val)
                        RSO.append(val)
                    if(val<70 ):
                        HOL1.append(val)
                        HOL1O.append(val)
                    if(val<54):
                        HOL2.append(val)
                        HOL2O.append(val)
                else:
                    if(val>180 ):
                        HP.append(val)
                        HPD.append(val) 
                    if(val>250 ):
                        HPC.append(val)
                        HPCD.append(val)
                    if(val>=70 and val<=180 ):
                        R.append(val)
                        RD.append(val)
                    if(val>=70 and val<=150 ):
                        RS.append(val)
                        RSD.append(val)
                    if(val<70 ):
                        HOL1.append(val)
                        HOL1D.append(val)
                    if(val<54 ):
                        HOL2.append(val)
                        HOL2D.append(val)
    return putval

def writeresultmanual(R1D, HP1D, HPC1D, HP1O, HOL21D, HP1, R1, HOL11O, HPC1, HOL11D, RS1, RS1D, R1O, HOL11, HOL21, RS1O, HPC1O, HOL21O):
        rows = [['Manual Mode',calc(HP1O),calc(HPC1O),calc(R1O),calc(RS1O),calc(HOL11O),calc(HOL21O),calc(HP1D),calc(HPC1D),calc(R1D),calc(RS1D),calc(HOL11D),calc(HOL21D),calc(HP1),calc(HPC1),calc(R1),calc(RS1),calc(HOL11),calc(HOL21)],
                ['Auto Mode',calc(List[0]),calc(List[1]),calc(List[2]),calc(List[3]),calc(List[4]),calc(List[5]),calc(List[6]),calc(List[7]),calc(List[8]),calc(List[9]),calc(List[10]),calc(List[11]),calc(List[12]),calc(List[13]),calc(List[14]),calc(List[15]),calc(List[16]),calc(List[17])]]
        fields = ['','Overnight Percentage time in hyperglycemia (CGM > 180 mg/dL)','Overnight percentage of time in hyperglycemia critical (CGM > 250 mg/dL)','Overnight percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)','Overnight percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)','Overnight percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)','Overnight percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)','Daytime Percentage time in hyperglycemia (CGM > 180 mg/dL)','Daytime percentage of time in hyperglycemia critical (CGM > 250 mg/dL)','Daytime percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)','Daytime percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)','Daytime percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)','Daytime percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)','Whole Day Percentage time in hyperglycemia (CGM > 180 mg/dL)','Whole day percentage of time in hyperglycemia critical (CGM > 250 mg/dL)','Whole day percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)','Whole day percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)','Whole day percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)','Whole Day percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)'] 
        with open(file +"\Results.csv", 'w') as csvfile: 
            csvwriter = csv.writer(csvfile) 
            csvwriter.writerow(fields)         
            csvwriter.writerows(rows)
        csvfile.close()

def storeresultauto(R1D, HP1D, HPC1D, HP1O, HOL21D, HP1, R1, HOL11O, HPC1, HOL11D, RS1, RS1D, R1O, HOL11, HOL21, RS1O, HPC1O, HOL21O):
        global List
        List1=[HP1O,HPC1O,R1O,RS1O,HOL11O,HOL21O,HP1D,HPC1D,R1D,RS1D,HOL11D,HOL21D,HP1,HPC1,R1,RS1,HOL11,HOL21]
        List=deepcopy(List1)
        
        

def invoke():
    HP1,HPC1,R1,RS1,HOL11,HOL21,HP,HPC,R,RS,HOL1,HOL2=([] for i in range(12))    
    return HP1,HPC1,R1,RS1,HOL11,HOL21,HP,HPC,R,RS,HOL1,HOL2

def invokeo():
    HP1O,HPC1O,R1O,RS1O,HOL11O,HOL21O,HPO,HPCO,RO,RSO,HOL1O,HOL2O=([] for i in range(12))
    return HP1O,HPC1O,R1O,RS1O,HOL11O,HOL21O,HPO,HPCO,RO,RSO,HOL1O,HOL2O
def invoked():
    HP1D,HPC1D,R1D,RS1D,HOL11D,HOL21D,HPD,HPCD,RD,RSD,HOL1D,HOL2D=([] for i in range(12))
    return HP1D,HPC1D,R1D,RS1D,HOL11D,HOL21D,HPD,HPCD,RD,RSD,HOL1D,HOL2D

def append(HP1,HPC1,R1,RS1,HOL11,HOL21,HP,HPC,R,RS,HOL1,HOL2):
    HP1.append((sum(HP)/288)*100)
    HPC1.append((sum(HPC)/288)*100)
    R1.append((sum(R)/288)*100)
    RS1.append((sum(RS)/288)*100)
    HOL11.append((sum(HOL1)/288)*100)
    HOL21.append((sum(HOL2)/288)*100)
      
def appendo(HP1O,HPC1O,R1O,RS1O,HOL11O,HOL21O,HPO,HPCO,RO,RSO,HOL1O,HOL2O):
    HP1O.append((sum(HPO)/288)*100)
    HPC1O.append((sum(HPCO)/288)*100)
    R1O.append((sum(RO)/288)*100)
    RS1O.append((sum(RSO)/288)*100)
    HOL11O.append((sum(HOL1O)/288)*100)
    HOL21O.append((sum(HOL2O)/288)*100)
    
def appendd(HP1D,HPC1D,R1D,RS1D,HOL11D,HOL21D,HPD,HPCD,RD,RSD,HOL1D,HOL2D):
    HP1D.append((sum(HPD)/288)*100)
    HPC1D.append((sum(HPCD)/288)*100)
    R1D.append((sum(RD)/288)*100)
    RS1D.append((sum(RSD)/288)*100)
    HOL11D.append((sum(HOL1D)/288)*100)
    HOL21D.append((sum(HOL2D)/288)*100)

def chkovernight(stamp):
    if(stamp.time()<datetime.time(6,0,0) or datetime.time(23,59,0)<stamp.time()):
        return True
    else:
        return False

def calc(val):
    x=ld.date()-fd.date()
    print(x)
    return sum(val)/(x.days+1)

    
def chkautomode(val):
    if(val<autodate):
        return True
    else:
        return False

autodatemethod()
processvals()




