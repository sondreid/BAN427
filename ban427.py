# BAN427 Insurance Analytics 

# Modules
import pandas as pd
import os
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection, metrics
from sklearn.metrics import confusion_matrix, accuracy_score
# Import data from excel to raw_df
master = pd.read_excel("exam_case_data.xlsx")
raw_df = master.copy()



# Adding new columns
raw_df['FULL_CHURN']    = np.where(raw_df['TIME2'] != 2, 1, 0)
raw_df['PARTIAL_CHURN'] = np.where((raw_df['NUMBER_COVERS_TIME2'] - raw_df['NUMBER_COVERS_TIME1']) < 0, 1, 0)
raw_df['MORE_SALE']     = np.where((raw_df['NUMBER_COVERS_TIME2'] - raw_df['NUMBER_COVERS_TIME1']) > 0, 1, 0)



# What characterizes more sale customers?
more_sale_customers = raw_df.loc[raw_df['MORE_SALE'] ==1]
non_more_sale_customers = raw_df.loc[raw_df['MORE_SALE'] ==0]


################ Clean data errors in tenure difference #############
df = raw_df[(raw_df['TENURE_TIME2'] - raw_df['TENURE_TIME1'] == 0.5) | (raw_df['TIME2']).isnull()]

df.loc[:,'NUMBER_COVERS_TIME2'] = df.loc[:,'NUMBER_COVERS_TIME2'].fillna(0)
df.loc[:,'TOTAL_PREM_TIME2']    = df.loc[:,'TOTAL_PREM_TIME2'].fillna(0)




# What characterizes more sale customers?
more_sale_customers = raw_df.loc[raw_df['MORE_SALE'] ==1]
non_more_sale_customers = raw_df.loc[raw_df['MORE_SALE'] ==0]


################ Clean data errors in tenure difference #############

df[(df['TENURE_TIME2'] - df['TENURE_TIME1'] == 0.5) | (df['TIME2']).isnull()]
df = raw_df[(raw_df['TENURE_TIME2'] - raw_df['TENURE_TIME1'] == 0.5) | (raw_df['TIME2']).isnull()]

########################### Descriptive statistics ------------------------------------------------------------------------------------------------------------------------------
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

################################### TABLES ##############################
### Age table ##

age_table = df.groupby(by=["AGE_GROUP"]).describe().loc[:,['FULL_CHURN','PARTIAL_CHURN', "MORE_SALE"]]

age_table.style



# Binary variables table

format_table_dict = {'Percentage': '{:.2%}'}
table_binary = pd.DataFrame([[(df.loc[df['FULL_CHURN'] == 1, 'FULL_CHURN']).count(), (df.loc[df['FULL_CHURN'] == 1, 'FULL_CHURN']).count()/len(df)],
                     [(df.loc[df['PARTIAL_CHURN'] == 1, 'PARTIAL_CHURN']).count(), (df.loc[df['PARTIAL_CHURN'] == 1, 'PARTIAL_CHURN']).count()/len(df)],
                     [(df.loc[df['MORE_SALE'] == 1, 'MORE_SALE']).count(), (df.loc[df['MORE_SALE'] == 1, 'MORE_SALE']).count()/len(df)]],
                     index = ['Full churn (Positive)', 'Partial churn (Positive)', 'More sales (Positive)'],
                     columns = ["Count", "Percentage"])

table_binary.style.format(format_table_dict)



## Number of covers table

table_continous = pd.DataFrame({'Tenure time 1':(df['TENURE_TIME1']).describe()[1:,], 'Tenure time 2':(df['TENURE_TIME2']).describe()[1:,], 
                              'Number of Covers in period 1': (df['NUMBER_COVERS_TIME1']).describe()[1:,],
                              'Number of Covers in period 2': (df['NUMBER_COVERS_TIME2']).describe()[1:,]})

table_continous.style.format('{:.2f}')





##  Churn and more sales by the size of portfolio. 


df['PREMIUM_INCREASE'] = np.where((df['TOTAL_PREM_TIME2'] - df['TOTAL_PREM_TIME1']) > 0, 1, 0)

df.groupby(by=["PREMIUM_INCREASE"]).describe().loc[:,['FULL_CHURN','PARTIAL_CHURN', "MORE_SALE"]]


(df['TOTAL_PREM_TIME2'] - df['TOTAL_PREM_TIME1']).describe()



##  Churn and more sales by whether customers has filed a claim
df.groupby(by=['CLAIM_EVENT_BEFORE_TIME1']).describe().loc[:,['FULL_CHURN','PARTIAL_CHURN', "MORE_SALE"]]


df.groupby(by=['AGE_GROUP', 'WOMAN']).describe().loc[:,['FULL_CHURN','PARTIAL_CHURN', "MORE_SALE"]]


df.groupby(by=['WOMAN', 'AGE_GROUP']).describe().loc[:,['FULL_CHURN','PARTIAL_CHURN', "MORE_SALE"]]

#### Signifiance tests


#stats.ttest_ind(df.loc[df['AGE'] < 30 & df.loc[df['WOMAN'] == 1], ['FULL_CHURN']], df.loc[df['AGE'] < 30 & df.loc[df['WOMAN'] == 0], ['FULL_CHURN']])

#stats.ttest_ind(df.loc[(df.AGE < 30) & (df.WOMAN == 1), 'FULL_CHURN'], df.loc[(df.AGE < 30) & (df.WOMAN == 0), 'FULL_CHURN'])


## For all FULL CHURN MEN VS WOMEN

age_woman = df.loc[df.WOMAN == 1, ['FULL_CHURN', 'PARTIAL_CHURN', 'MORE_SALE', 'AGE_GROUP']].groupby(by = 'AGE_GROUP').describe()
age_men   = df.loc[df.WOMAN == 0, ['FULL_CHURN', 'PARTIAL_CHURN', 'MORE_SALE', 'AGE_GROUP']].groupby(by = 'AGE_GROUP').describe()

stats.ttest_ind(age_woman["FULL_CHURN"]['mean'], age_men["FULL_CHURN"]["mean"], equal_var=False)


################################### PLOTS ##############################

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

# MORE SALE
hist_df_more_sale = df.loc[:,['MORE_SALE','AGE_GROUP', 'WOMAN']].groupby(by = ["AGE_GROUP", 'WOMAN']).sum()
hist_df_more_sale.index.name = 'AGE_GROUP'
hist_df_more_sale.reset_index(inplace=True)

# More sales plot
bar_plot(hist_df_more_sale, "AGE_GROUP", 'WOMAN', 'MORE_SALE', 'GENDER', 'Men', 'Women', 'Age Group', 'Number of increase in coverage' )





####################### Prediction model #########################

# Creating features and prediction variables
x = df.loc[:, ~df.columns.isin(['TIME1', 'TIME2', 'TENURE_TIME2', 'AVERAGE_INCOME_COUNTY_TIME1','FULL_CHURN', 'PARTIAL_CHURN', 'MORE_SALE'])]

y_full_churn    = df['FULL_CHURN']
y_partial_churn = df['PARTIAL_CHURN']
y_more_sale     = df['MORE_SALE']


# Splitting the data into train and test [fc = full churn, pc = partial churn, ms = more sales]
from sklearn.model_selection import train_test_split

xtrain_fc, xtest_fc, ytrain_fc, ytest_fc = train_test_split(x, y_full_churn,    test_size = 0.2, random_state = 0)
xtrain_pc, xtest_pc, ytrain_pc, ytest_pc = train_test_split(x, y_partial_churn, test_size = 0.2, random_state = 0)
xtrain_ms, xtest_ms, ytrain_ms, ytest_ms = train_test_split(x, y_more_sale,     test_size = 0.2, random_state = 0)


# Scaling the features
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

xtrain_fc = sc.fit_transform(xtrain_fc)
xtest_fc  = sc.transform(xtest_fc)

xtrain_pc = sc.fit_transform(xtrain_pc)
xtest_pc  = sc.transform(xtest_pc)

xtrain_ms = sc.fit_transform(xtrain_ms)
xtest_ms  = sc.transform(xtest_ms)


########################## Training the logistic regression model
from sklearn.linear_model import LogisticRegression

logreg_fc = LogisticRegression(random_state = 0)
logreg_fc.fit(xtrain_fc, ytrain_fc)

logreg_pc = LogisticRegression(random_state = 0)
logreg_pc.fit(xtrain_pc, ytrain_pc)

logreg_ms = LogisticRegression(random_state = 0)
logreg_ms.fit(xtrain_ms, ytrain_ms)


# Predicting the logreg model
ypred_logreg_fc = logreg_fc.predict(xtest_fc)
yprob_logreg_fc = (logreg_fc.predict_proba(xtest_fc)[:,1]  >= 0.05).astype(bool)

ypred_logreg_pc = logreg_pc.predict(xtest_pc)
yprob_logreg_pc = (logreg_pc.predict_proba(xtest_pc)[:,1]  >= 0.05).astype(bool)

ypred_logreg_ms = logreg_ms.predict(xtest_ms)
yprob_logreg_ms = (logreg_pc.predict_proba(xtest_ms)[:,1]  >= 0.05).astype(bool)



cm_fc = confusion_matrix(ytest_fc, ypred_logreg_fc)
print(cm_fc)
accuracy_score(ytest_fc, ypred_logreg_fc)

cm_pc = confusion_matrix(ytest_pc, ypred_logreg_pc)
print(cm_pc)
accuracy_score(ytest_pc, ypred_logreg_pc)

cm_ms = confusion_matrix(ytest_ms, ypred_logreg_ms)
print(cm_ms)
accuracy_score(ytest_ms, ypred_logreg_ms)

### ROC-curve logreg ###

roc_logreg_fc = roc(ytrain_fc, ytest_fc, yprob_logreg_fc)
roc_logreg_fc

roc_logreg_pc = roc(ytrain_pc, ytest_pc, yprob_logreg_pc)
roc_logreg_pc

roc_logreg_ms = roc(ytrain_ms, ytest_ms, yprob_logreg_ms)
roc_logreg_ms


########################## Training the KNN-model
from sklearn.neighbors import KNeighborsClassifier

knn_fc = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_fc.fit(xtrain_fc, ytrain_fc)

knn_pc = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_pc.fit(xtrain_pc, ytrain_pc)

knn_ms = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_ms.fit(xtrain_ms, ytrain_ms)

# Predicting the knn models
ypred_knn_fc = knn_fc.predict(xtest_fc)
yprob_knn_fc = (knn_fc.predict_proba(xtest_fc)[:,1]  >= 0.05).astype(bool)

ypred_knn_pc = knn_pc.predict(xtest_pc)
yprob_knn_pc = (knn_pc.predict_proba(xtest_pc)[:,1]  >= 0.05).astype(bool)

ypred_knn_ms = knn_ms.predict(xtest_ms)
yprob_knn_ms = (knn_ms.predict_proba(xtest_ms)[:,1]  >= 0.05).astype(bool)


# Checking the accuracy with confusion matrix
cm_knn_fc = confusion_matrix(ytest_fc, ypred_knn_fc)
print(cm_knn_fc)
accuracy_score(ytest_fc, ypred_knn_fc)


cm_knn_pc = confusion_matrix(ytest_pc, ypred_knn_pc)
print(cm_knn_pc)
accuracy_score(ytest_pc, ypred_knn_pc)


cm_knn_ms = confusion_matrix(ytest_ms, ypred_knn_ms)
print(cm_knn_ms)
accuracy_score(ytest_ms, ypred_knn_ms)


### ROC-curve KNN ###

roc_KNN_fc = roc(ytrain_fc, ytest_fc, yprob_knn_fc)
roc_KNN_fc

roc_KNN_pc = roc(ytrain_pc, ytest_pc, yprob_knn_pc)
roc_KNN_pc

roc_KNN_ms = roc(ytrain_ms, ytest_ms, yprob_knn_ms)
roc_KNN_ms


########################## Training the SVM-model


svc_fc = SVC(kernel = 'linear', random_state = 0, probability = True)
svc_fc.fit(xtrain_fc, ytrain_fc)

svc_pc = SVC(kernel = 'linear', random_state = 0, probability = True)
svc_pc.fit(xtrain_pc, ytrain_pc)

svc_ms = SVC(kernel = 'linear', random_state = 0, probability = True)
svc_ms.fit(xtrain_ms, ytrain_ms)

# Predicting the svm model
ypred_svc_fc = (svc_fc.predict_proba(xtest_fc)[:,1]  >= 0.05).astype(bool)
ypred_svc_pc = (svc_pc.predict_proba(xtest_pc)[:,1]  >= 0.05).astype(bool)
ypred_svc_ms = (svc_ms.predict_proba(xtest_ms)[:,1]  >= 0.05).astype(bool)

# Checking the accuracy with confusion matrix
cm_svc_fc = confusion_matrix(ytest_fc, ypred_svc_fc)
print(cm_svc_fc)
accuracy_score(ytest_fc, ypred_svc_fc)

cm_svc_pc = confusion_matrix(ytest_pc, ypred_svc_pc)
print(cm_svc_pc)
accuracy_score(ytest_pc, ypred_svc_pc)

cm_svc_ms = confusion_matrix(ytest_ms, ypred_svc_ms)
print(cm_svc_ms)
accuracy_score(ytest_ms, ypred_svc_ms)


###################### ROC Curve ################################### 


def roc(ytrain, ytest, ypred):
    """
    
    """
    fpr, tpr, tr = metrics.roc_curve(ytest, ypred[:,1])
    auc = metrics.roc_auc_score(ytest, ypred[:, 1])

    fpr1, tpr1, tr = metrics.roc_curve(ytrain, ypred[:,1])
    auc1 = metrics.roc_auc_score(ytrain, ypred[:,1])

    plt.figure(num = None, figsize = (10,10), dpi = 80)
    plt.plot(fpr, tpr, label = 'SVM test data (area = %0.2f)' % auc)
    plt.plot(fpr1, tpr1, label = 'SVM train data (area = %0.2f)' % auc1)
    plt.plot((0,1), (1,0), ls = "--", c = ".3")
    plt.title = (' ROC Curve - test and train data')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend()
    plt.show()
    return plt


roc_SVM_fc = roc(ytrain_fc, ytest_fc, ypred_svc_fc)
roc_SVM_fc

roc_SVM_pc = roc(ytrain_pc, ytest_pc, ypred_svc_pc)
roc_SVM_pc

roc_SVM_ms = roc(ytrain_ms, ytest_ms, ypred_svc_ms)
roc_SVM_ms




########################## Training the Naive bayes-model

nb_fc = GaussianNB
nb_fc.fit(xtrain_fc, ytrain_fc)

