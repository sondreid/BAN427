# BAN427 Insurance Analytics 

# Modules
import pandas as pd
import os
import numpy as np

# Import data from excel to df
master = pd.read_excel("exam_case_data.xlsx")
df = master.copy()


# New columns
df['FULL_CHURN']    = np.where(df['TIME2'] != 2, 1, 0)
df['PARTIAL_CHURN'] = np.where((df['NUMBER_COVERS_TIME2'] - df['NUMBER_COVERS_TIME1']) < 0, 1, 0)
df['MORE_SALE']     = np.where((df['NUMBER_COVERS_TIME2'] - df['NUMBER_COVERS_TIME1']) > 0, 1, 0)



# What characterizes more sale customers?
more_sale_customers = df.loc[df['MORE_SALE'] ==1]


# Descriptive statistics
df = pd.DataFrame(df)
df.FULL_CHURN.astype('category').describe()
df.PARTIAL_CHURN.astype('category').describe()
df.MORE_SALE.astype('category').describe()

# Descriptive statistics with group by AGE_GROUP
def age_groups(x):
    if   x < 30:
        return '<30'
    elif x < 40:
        return '<40'
    elif x < 50:
        return '<50'
    elif x < 60:
        return '<60'
    elif x < 70:
        return '<70'
    else:
        return '>=70'

df['AGE_GROUP'] = df['AGE'].apply(age_groups)

df.groupby(by=["AGE_GROUP"]).describe().loc[:,['FULL_CHURN','PARTIAL_CHURN', "MORE_SALE"]]


((df.TENURE_TIME2 - df.TENURE_TIME1) >20).sum()

df.loc[df['TENURE_TIME2']- df['TENURE_TIME1'] < 0]



##  Churn and more sales by the size of portfolio. 


df['PREMIUM_INCREASE'] = np.where((df['TOTAL_PREM_TIME2'] - df['TOTAL_PREM_TIME1']) > 0, 1, 0)

df.groupby(by=["PREMIUM_INCREASE"]).describe().loc[:,['FULL_CHURN','PARTIAL_CHURN', "MORE_SALE"]]


(df['TOTAL_PREM_TIME2'] - df['TOTAL_PREM_TIME1']).describe()


def size_groups(x):
    """'
    """

    if   x < 30:
        return '<30'
    elif x < 40:
        return '<40'
    elif x < 50:
        return '<50'
    elif x < 60:
        return '<60'
    elif x < 70:
        return '<70'
    else:
        return '>=70'


col(df)

##  Churn and more sales by whether customers has filed a claim
df.groupby(by=['CLAIM_EVENT_BEFORE_TIME1']).describe().loc[:,['FULL_CHURN','PARTIAL_CHURN', "MORE_SALE"]]
