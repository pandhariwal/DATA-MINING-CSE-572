import pandas as pd

data=pd.read_csv("E:\photocopy\Vinay Pandhariwal\COLLEGE\ASU\Courses\DM\Assignment1\CGMData.csv")

HP=0
HPC=0
R=0
RS=0
HOL1=0
HOL2=0

for row in data['Sensor Glucose (mg/dL)'].iteritems():
        if(row):
            if(row>180 and row<=250):
                HP += 1
            if(row>250):
                HPC += 1
            if(row>=70 and row<=180):
                R += 1
            if(row>=70 and row<=150):
                RS += 1
            if(row<70 and row>=54):
                HOL1 += 1
            if(row<54):
                HOL2 += 1
print(R)