import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import os
os.chdir("D:\Imarticus\Python_program")
os.getcwd()

bankset = pd.read_csv("BankCreditCard.csv")

from sklearn.model_selection import train_test_split
Trainset, Testset = train_test_split(bankset, test_size = 0.3)

Trainset["Source"] = "Train"
Testset["Source"] = "Test"

fullraw = pd.concat([Trainset,Testset], axis = 0)

fullraw.isnull().sum()

fullraw.loc[fullraw["Source"] == "Train","Default_Payment"].value_counts()\
                    /fullraw.loc[fullraw['Source']=='Train'].shape[0]

fullraw.drop(["Customer ID"], axis=1, inplace=True)

import numpy as np

cond = [fullraw['Gender'] == 1, fullraw['Gender']==2]
choice = ["Male","Female"]
fullraw['Gender']= np.select(cond,choice)

fullraw['Marital'].unique
cond = [fullraw['Marital'] == 0,fullraw['Marital'] == 1, fullraw['Marital']==2, fullraw['Marital'] == 3]
choice = ["unknown","Single","Married","Not Interested"]
fullraw['Marital']= np.select(cond,choice)

cond = [fullraw['Academic_Qualification'] == 1, fullraw['Academic_Qualification']==2,
        fullraw['Academic_Qualification']==3,fullraw['Academic_Qualification']==4,
        fullraw['Academic_Qualification']==5,fullraw['Academic_Qualification']==6]
choice = ["Under Graduate","Graduate","Post Graduate","Professional","Others","Unknown"]
fullraw['Academic_Qualification']= np.select(cond,choice)


fullraw1=pd.get_dummies(fullraw,drop_first=True)

fullraw1["Intercept"]= 1

Train = fullraw1[fullraw1['Source_Train'] == 1].drop(['Source_Train'], axis = 1).copy()
Test = fullraw1[fullraw1['Source_Train'] == 0].drop(['Source_Train'], axis = 1).copy()


Train, Test = train_test_split(fullraw1, train_size = 0.7, random_state=123)

Train_x = Train.drop(['Default_Payment'], axis = 1).copy()
Test_x = Test.drop(['Default_Payment'], axis = 1).copy()
Train_y = Train['Default_Payment'].copy()
Test_y = Test['Default_Payment'].copy()



from statsmodels.stats.outliers_influence import variance_inflation_factor

Max_VIF = 10
Train_X_Copy = Train_x.copy()
counter = 1
High_VIF_Column_Names = []


while (Max_VIF >= 10):
    
    print(counter)
    
    VIF_Df = pd.DataFrame()   
    VIF_Df['VIF'] = [variance_inflation_factor(Train_X_Copy.values, i) for i in range(Train_X_Copy.shape[1])]  
    VIF_Df['Column_Name'] = Train_X_Copy.columns
    
    Max_VIF = max(VIF_Df['VIF'])
    Temp_Column_Name = VIF_Df.loc[VIF_Df['VIF'] == Max_VIF, 'Column_Name']
    print(Temp_Column_Name, ": ", Max_VIF)
    
    if (Max_VIF >= 10): # This condition will ensure that ONLY columns having VIF lower than 10 are NOT dropped
        print(Temp_Column_Name, Max_VIF)
        Train_X_Copy = Train_X_Copy.drop(Temp_Column_Name, axis = 1)    
        High_VIF_Column_Names.extend(Temp_Column_Name)
    
    counter = counter + 1

High_VIF_Column_Names.remove("Intercept")

Train_x.drop(['May_Bill_Amount','March_Bill_Amount'], axis = 1, inplace=True)
Test_x.drop(['May_Bill_Amount','March_Bill_Amount'], axis = 1, inplace=True)

Train_x.shape
Test_x.shape

from statsmodels.api import Logit
Model1 = Logit(Train_y, Train_x).fit()
Model1.summary()

col_names = ['Marital_Not Interested']
Model2 = Logit(Train_y,Train_x.drop( col_names, axis= 1)).fit()
Model2.summary()

col_names.append('June_Bill_Amount')
Model3 = Logit(Train_y,Train_x.drop( col_names, axis= 1)).fit()
Model3.summary()

col_names.append('Academic_Qualification_Under Graduate')
Model4 = Logit(Train_y,Train_x.drop( col_names, axis= 1)).fit()
Model4.summary()

col_names.append('Repayment_Status_Feb')
Model5 = Logit(Train_y,Train_x.drop(col_names, axis= 1)).fit()
Model5.summary()

Train_x = Train_x.drop(col_names, axis = 1)
Test_x = Test_x.drop(col_names, axis = 1)

Test_x['Predit'] = Model5.predict(Test_x)
Test_x.columns
Test_x['Predit'][0:6]

Test_x['Test_class']=np.where(Test_x['Predit']>=0.5, 1, 0)

import pandas as pd
confusion_matrix = pd.crosstab(Test_x['Test_class'], Test_y)

sum(np.diagonal(confusion_matrix))/Test_x.shape[0]*100

from sklearn.metrics import f1_score, precision_score, recall_score
f1_score(Test_y,Test_x['Test_class'])  #0.44
precision_score(Test_y,Test_x['Test_class'])  #0.70
recall_score(Test_y,Test_x['Test_class']) #0.33

from sklearn.metrics import roc_curve, auc

Train_prob = Model5.predict(Train_x)
fpr,tpr,cutoff = roc_curve(Train_y,Train_prob)

cutoff_table = pd.DataFrame()
cutoff_table['FPR'] = fpr
cutoff_table['TPR'] = tpr
cutoff_table['Cutoff'] = cutoff

import seaborn as sns

sns.lineplot(cutoff_table['FPR'], cutoff_table['TPR'])

auc(fpr,tpr)

cutoff_table['Difference'] = cutoff_table['TPR'] - cutoff_table['FPR']
max(cutoff_table['Difference'])

import numpy as np
cutoff_table['Distance'] = np.sqrt((1-cutoff_table['TPR'])**2 + (0-cutoff_table['FPR']**2))
min(cutoff_table['Distance'])

#### Based of Max difference 

Test_x['Test_class1']=np.where(Test_x['Predit']>=0.2, 1, 0)

confusion_matrix1 = pd.crosstab(Test_x['Test_class1'], Test_y)


sum(np.diagonal(confusion_matrix))/Test_x.shape[0]*100

from sklearn.metrics import f1_score, precision_score, recall_score
f1_score(Test_y,Test_x['Test_class1'])  #0.51
precision_score(Test_y,Test_x['Test_class1'])  #0.45
recall_score(Test_y,Test_x['Test_class1']) #0.60

auc(fpr,tpr)

#### Based on Euclidean distance - should consider minimum distance.

Test_x['Test_class2']=np.where(Test_x['Predit']>=0.16, 1, 0)

confusion_matrix2 = pd.crosstab(Test_x['Test_class2'], Test_y)


sum(np.diagonal(confusion_matrix))/Test_x.shape[0]*100

from sklearn.metrics import f1_score, precision_score, recall_score
f1_score(Test_y,Test_x['Test_class2'])  #0.48
precision_score(Test_y,Test_x['Test_class2'])  #0.36
recall_score(Test_y,Test_x['Test_class2']) #0.73

auc(fpr,tpr)  #76

######## Satsified Sampling 

#pip install imblearn
from imblearn.under_sampling import RandomUnderSampler

Train_y.value_counts()

RUS = RandomUnderSampler(ratio=0.7, random_state=123)

Train_x_RUS, Train_y_RUS = RUS.fit_sample(Train_x, Train_y)
Train_x_RUS= pd.DataFrame(Train_x_RUS)
Train_y_RUS = pd.Series(Train_y_RUS)

Train_x_RUS.columns=Train_x.columns
Train_y_RUS.name = Train_y.name

Train_y_RUS.value_counts()
Train_y_RUS.value_counts()[1]/sum(Train_y_RUS.value_counts())*100

from statsmodels.api import Logit
Model6 = Logit(Train_y_RUS, Train_x_RUS).fit()
Model1.summary()

Test_X = Test_x.copy()

Test_X.drop(['Predit'], axis = 1, inplace= True)
Test_X['Predit'] = Model6.predict(Test_X)

Test_X['Test_Class_RUS'] = np.where(Test_X['Predit']>=0.5, 1,0)

confusion_matrix2 = pd.crosstab(Test_X['Test_Class_RUS'], Test_y)


sum(np.diagonal(confusion_matrix2))/Test_X.shape[0]*100

from sklearn.metrics import f1_score, precision_score, recall_score
f1_score(Test_y,Test_X['Test_Class_RUS'])  #0.53
precision_score(Test_y,Test_X['Test_Class_RUS'])  #0.55
recall_score(Test_y,Test_X['Test_Class_RUS']) #0.51

auc(fpr,tpr)  #76
