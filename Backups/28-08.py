# BAN427 Insurance Analytics 

# Modules
import pandas as pd
import os
import numpy as np
from scipy.stats import ttest_ind
import seaborn as sns


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



##  Churn and more sales by whether customers has filed a claim
df.groupby(by=['CLAIM_EVENT_BEFORE_TIME1']).describe().loc[:,['FULL_CHURN','PARTIAL_CHURN', "MORE_SALE"]]

age_woman = df.groupby(by=['AGE_GROUP', 'WOMAN']).describe().loc[:,['FULL_CHURN','PARTIAL_CHURN', "MORE_SALE"]]
age_woman

age_woman = df.loc[df.WOMAN == 1, ['FULL_CHURN', 'PARTIAL_CHURN', 'MORE_SALE', 'AGE_GROUP']].groupby(by = 'AGE_GROUP').describe()
age_men = df.loc[df.WOMAN == 0, ['FULL_CHURN', 'PARTIAL_CHURN', 'MORE_SALE', 'AGE_GROUP']].groupby(by = 'AGE_GROUP').describe()




ttest_ind(age_woman["FULL_CHURN"]['mean'], age_men["FULL_CHURN"]["mean"], equal_var=False)


df.loc[:,['FULL_CHURN','AGE_GROUP']]

hist_df = [['AGE_GROUP' , df['AGE_GROUP']], ['COUNT',df.loc[:,['FULL_CHURN','AGE_GROUP']].groupby(by = "AGE_GROUP").count()]]

hist_df = [['AGE_GROUP' , df['AGE_GROUP']], ['COUNT',df.loc[:,['FULL_CHURN','AGE_GROUP']].groupby(by = "AGE_GROUP").count()]]

(df['AGE_GROUP']).iloc[0:10]

hist_df_list = [df['AGE_GROUP'], df.loc[:,['FULL_CHURN','AGE_GROUP']].groupby(by = "AGE_GROUP").count()]

pd.unique(df['AGE_GROUP'])


df.loc[:,['FULL_CHURN','AGE_GROUP']].groupby(by = "AGE_GROUP").count()['FULL_CHURN']

age_series = pd.Series(pd.unique(df['AGE_GROUP']))
full_churn_series = pd.Series(df.loc[:,['FULL_CHURN','AGE_GROUP']].groupby(by = "AGE_GROUP").count()['FULL_CHURN'])


hist_df = df.loc[:,['FULL_CHURN','AGE_GROUP']].groupby(by = "AGE_GROUP").count()
hist_df.index.name = 'AGE_GROUP'
hist_df.reset_index(inplace=True)
sns.barplot(x = hist_df['AGE_GROUP'], y = hist_df['FULL_CHURN'])






hist_df = df.loc[:,['FULL_CHURN','AGE_GROUP', 'WOMAN']].groupby(by = ["AGE_GROUP", 'WOMAN']).count()
hist_df.index.name = 'AGE_GROUP'
hist_df.reset_index(inplace=True)
sns.set_theme(style = "whitegrid")
sns.barplot(x = 'AGE_GROUP', y = 'FULL_CHURN', data = hist_df, hue = 'WOMAN')