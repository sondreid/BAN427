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



sns.set_theme(palette='pastel')
def bar_plot(df, x_var, hue_var, y_var, label_title, x_label, y_label, x_axis_label, y_axis_label):
    """
    Generates a bar plot with hue.
    
    Parameters:
        df: input dataframe
        x_var : x variable
        hue_var: category variable. Is left out if empty string "". 
        y_var: y variable
        label_title: Title of categories
        x_label:

    """ 
    if hue_var == "":
        ax = sns.barplot(data = df,
                x = x_var, 
                y = y_var)
        ax.set_ylabel(y_axis_label)
        ax.set_xlabel(x_axis_label)
    else: 
        ax = sns.barplot(data = df,
                    x = x_var, 
                    y = y_var, 
                    hue = hue_var)
        ax.set_ylabel(y_axis_label)
        ax.set_xlabel(x_axis_label)
        labels = [x_label, y_label]
        h, l = ax.get_legend_handles_labels()
        ax.legend(h, labels, title = label_title)
    return ax

# Line plot full churn by age group:
df.FULL_CHURN.groupby(df.AGE_GROUP).count().plot()


# FULL CHURNERS
hist_df_full_churn_no_claim = df.loc[df.CLAIM_EVENT_BEFORE_TIME1 == 0,['FULL_CHURN','AGE_GROUP', 'WOMAN']].groupby(by = ["AGE_GROUP", 'WOMAN']).sum()
hist_df_full_churn_no_claim.index.name = 'AGE_GROUP'
hist_df_full_churn_no_claim.reset_index(inplace=True)

bar_plot(hist_df_full_churn_no_claim, "AGE_GROUP", 'WOMAN', 'FULL_CHURN', 'GENDER', 'Men', 'Women', 'Claim Event', 'Number of full churners')

# FULL CHURNERS given claim
hist_df_full_churn_given_claim = df.loc[df.CLAIM_EVENT_BEFORE_TIME1 == 1,['FULL_CHURN','AGE_GROUP', 'WOMAN']].groupby(by = ["AGE_GROUP", 'WOMAN']).sum()
hist_df_full_churn_given_claim.index.name = 'AGE_GROUP'
hist_df_full_churn_given_claim.reset_index(inplace=True)

bar_plot(hist_df_full_churn_given_claim, "AGE_GROUP", 'WOMAN', 'FULL_CHURN', 'GENDER', 'Men', 'Women', 'Claim Event', 'Number of full churners given filed claim' )



# FULL CHURN by claim event 
hist_df_full_claim_event = df.loc[:,['FULL_CHURN','CLAIM_EVENT_BEFORE_TIME1', 'WOMAN']].groupby(by = ['CLAIM_EVENT_BEFORE_TIME1', 'WOMAN']).mean()
hist_df_full_claim_event.index.name = 'CLAIM_EVENT_BEFORE_TIME1'
hist_df_full_claim_event.reset_index(inplace=True)

bar_plot(hist_df_full_claim_event, "CLAIM_EVENT_BEFORE_TIME1", 'WOMAN', 'FULL_CHURN', 'GENDER', 'Men', 'Women', 'Claim Event', 'Number of full churners' )




# PARTIAL CHURNERS
hist_df_partial_churn = df.loc[:,['PARTIAL_CHURN','AGE_GROUP', 'WOMAN']].groupby(by = ["AGE_GROUP", 'WOMAN']).sum()
hist_df_partial_churn.index.name = 'AGE_GROUP'
hist_df_partial_churn.reset_index(inplace=True)

# Partial plot
bar_plot(hist_df_partial_churn, "AGE_GROUP", 'WOMAN', 'PARTIAL_CHURN', 'GENDER', 'Men', 'Women', 'Age Group', 'Number of partial churners' )


# MORE SALE
hist_df_more_sale = df.loc[:,['MORE_SALE','AGE_GROUP', 'WOMAN']].groupby(by = ["AGE_GROUP", 'WOMAN']).sum()
hist_df_more_sale.index.name = 'AGE_GROUP'
hist_df_more_sale.reset_index(inplace=True)

# More sales plot
bar_plot(hist_df_more_sale, "AGE_GROUP", 'WOMAN', 'MORE_SALE', 'GENDER', 'Men', 'Women', 'Age Group', 'Number of increase in coverage' )




