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



# Descriptive statistics
df = pd.DataFrame(df)
df.FULL_CHURN.astype('category').describe()
df.PARTIAL_CHURN.astype('category').describe()
df.MORE_SALE.astype('category').describe()







