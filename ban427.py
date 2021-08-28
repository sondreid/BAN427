# BAN427 Insurance Analytics 

# Modules
import pandas as pd
import os
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# Import data from excel to df
master = pd.read_excel("exam_case_data.xlsx")
df = master.copy()


# New columns
df['FULL_CHURN']    = np.where(df['TIME2'] != 2, 1, 0)
df['PARTIAL_CHURN'] = np.where((df['NUMBER_COVERS_TIME2'] - df['NUMBER_COVERS_TIME1']) < 0, 1, 0)
df['MORE_SALE']     = np.where((df['NUMBER_COVERS_TIME2'] - df['NUMBER_COVERS_TIME1']) > 0, 1, 0)



# What characterizes more sale customers?
more_sale_customers = df.loc[df['MORE_SALE'] ==1]
non_more_sale_customers = df.loc[df['MORE_SALE'] ==0]




# Descriptive statistics
df = pd.DataFrame(df)
df.FULL_CHURN.astype('category').describe()
df.PARTIAL_CHURN.astype('category').describe()
df.MORE_SALE.astype('category').describe()

##  Churn and more sales by age groups. 
def age_groups(x):
    """'
    Function that outputs  a string denoting an agegroup depending on
    the input integer. 
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


df.groupby(by=['AGE_GROUP', 'WOMAN']).describe().loc[:,['FULL_CHURN','PARTIAL_CHURN', "MORE_SALE"]]


df.groupby(by=['WOMAN', 'AGE_GROUP']).describe().loc[:,['FULL_CHURN','PARTIAL_CHURN', "MORE_SALE"]]

#### Signifiance tests


stats.ttest_ind(df.loc[df['AGE'] < 30 & df.loc[df['WOMAN'] == 1], ['FULL_CHURN']], df.loc[df['AGE'] < 30 & df.loc[df['WOMAN'] == 0], ['FULL_CHURN']])

stats.ttest_ind(df.loc[(df.AGE < 30) & (df.WOMAN == 1), 'FULL_CHURN'], df.loc[(df.AGE < 30) & (df.WOMAN == 0), 'FULL_CHURN'])


## For all FULL CHURN MEN VS WOMEN

age_woman = df.loc[df.WOMAN == 1, ['FULL_CHURN', 'PARTIAL_CHURN', 'MORE_SALE', 'AGE_GROUP']].groupby(by = 'AGE_GROUP').describe()
age_men   = df.loc[df.WOMAN == 0, ['FULL_CHURN', 'PARTIAL_CHURN', 'MORE_SALE', 'AGE_GROUP']].groupby(by = 'AGE_GROUP').describe()

ttest_ind(age_woman["FULL_CHURN"]['mean'], age_men["FULL_CHURN"]["mean"], equal_var=False)



df.loc[:,['FULL_CHURN','AGE_GROUP']].groupby(by = "AGE_GROUP").count().plot()

hist_df = df['AGE_GROUP']

df.loc[:,['FULL_CHURN','AGE_GROUP']].groupby(by = "AGE_GROUP").count()


hist_df_list = [['AGE_GROUP' , df['AGE_GROUP']], ['COUNT',df.loc[:,['FULL_CHURN','AGE_GROUP']].groupby(by = "AGE_GROUP").count()]]

hist_df = pd.DataFrame(hist_df_list, columns = ['AGE_GROUP', 'COUNT'])

hist_df = df.loc[:,['FULL_CHURN','AGE_GROUP']].groupby(by = "AGE_GROUP").count()
hist_df.index.name = 'AGE_GROUP'
hist_df.reset_index(inplace=True)

hist_df

# Descriptive statistics plot
sns.histplot(data = hist_df, x = "AGE_GROUP", y = 'FULL_CHURN')




agegroup = df.loc[:,['FULL_CHURN','AGE_GROUP']].groupby(by = "AGE_GROUP").count()


type(agegroup)

agegroup.index

sns.histplot(data = agegroup, x = 'AGE_GROUP', y = 'FULL_CHURN')


df.CHURN_30.groupby(df.TEN_YEARS_CAT).count().plot()


df.FULL_CHURN.plot(kind = 'hist')

df.AGE_GROUP.hist(by = df.FULL_CHURN)



# SET SNS Them

sns.set_theme(palette='pastel')


# Plot full churn by age group:
df.FULL_CHURN.groupby(df.AGE_GROUP).count().plot()

hist_df = df.loc[:,['FULL_CHURN','AGE_GROUP', 'WOMAN']].groupby(by = ["AGE_GROUP", 'WOMAN']).count()
hist_df.index.name = 'AGE_GROUP'
hist_df.reset_index(inplace=True)


sns.barplot(data = hist_df,
            x = 'AGE_GROUP', 
            y = 'FULL_CHURN', 
            hue = 'WOMAN',
            xlabel = 'Full churn')
