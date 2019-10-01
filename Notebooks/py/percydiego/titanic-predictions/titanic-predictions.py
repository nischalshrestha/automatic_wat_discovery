#!/usr/bin/env python
# coding: utf-8

#  # Predicción de supervivientes del TITANIC

# In[ ]:


#Importando librerías

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import seaborn as sns

get_ipython().magic(u'matplotlib inline')
warnings.filterwarnings('ignore')
print(os.listdir('../input'))
pd.options.display.max_columns = 100


# In[ ]:


#Importando los datasets

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

print(train.shape)
print(test.shape)
features=train.columns.tolist()


# In[ ]:


df=pd.concat([train,test],)
df=df[features]
df.shape


# In[ ]:


df.set_index('PassengerId',inplace=True)
df.reset_index(inplace=True)
df.head()


# # Feature Engineering

# In[ ]:


df['Title']=df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
# normalizando titulos
normalized_titles = {
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"
}


# In[ ]:


# map the normalized titles to the current titles 
df['Title'] = df['Title'].map(normalized_titles)

# view value counts for the normalized titles
print(df.Title.value_counts())


# In[ ]:


df['Sex']=df['Sex'].map({'female':0, 'male':1})
df['Family_size']=df['SibSp'] + df['Parch']+1
df['Alone']=np.where(df['Family_size']==1,1,0)


    
    # introducing other features based on the family size
df['Singleton'] = df['Family_size'].map(lambda s: 1 if s == 1 else 0)
df['SmallFamily'] = df['Family_size'].map(lambda s: 1 if 2 <= s <= 4 else 0)
df['LargeFamily'] = df['Family_size'].map(lambda s: 1 if 5 <= s else 0)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
Var=['Sex','Family_size','Alone','Pclass','Age']

age_nomissing=df.loc[df.Age.notnull(),Var]
age_missing=df.loc[df.Age.isnull(),Var].drop('Age',axis=1).values
rf = RandomForestRegressor()

x=age_nomissing.drop('Age', axis=1).values
y=age_nomissing['Age'].values


rf.fit(x,y)


Age_predict=rf.predict(age_missing)

df.loc[df.Age.isnull(), 'Age' ] = Age_predict


# In[ ]:


df['Cabin']=df['Cabin'].str[0]
df['Cabin']=df['Cabin'].fillna('U')

df['Cabin']=df['Cabin'].replace(['A','G','T'],'U')
df['Cabin']=df['Cabin'].replace(['A','G','T'],'U')


# In[ ]:


def cleanTicket(ticket):
    ticket = ticket.replace('.', '')
    ticket = ticket.replace('/', '')
    ticket = ticket.split()
    ticket = map(lambda t : t.strip(), ticket)
    ticket = list(filter(lambda t : not t.isdigit(), ticket))
    if len(ticket) > 0:
        return ticket[0]
    else: 
        return 'XXX'



# In[ ]:


tickets = set()
for t in df['Ticket']:
    tickets.add(cleanTicket(t))   


# In[ ]:


df['Ticket']=df['Ticket'].apply(lambda x: cleanTicket(x))


# In[ ]:


df['Cabin'].value_counts()


# In[ ]:


df.head()


# In[ ]:


df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Fare']=df['Fare'].fillna(df['Fare'].median())


# In[ ]:


df.apply(lambda x: sum(x.isnull())).sort_values(ascending=False)


# In[ ]:


df['Age_range']=pd.factorize(pd.qcut(df.Age,4),sort=True)[0]


# In[ ]:


from sklearn.tree import DecisionTreeClassifier, export_graphviz
Variable=['Fare']

df_1=df.loc[df.Survived.notnull()]

tree=DecisionTreeClassifier(max_depth=3)
tree.fit(df_1[Variable],df_1['Survived'])

with open('tree.dot','w') as dotfile:
    export_graphviz(tree,out_file=dotfile,feature_names=Variable,filled=True)
    dotfile.close()

    
from graphviz import Source

with open('tree.dot','r') as f:
    text=f.read()
    plot=Source(text)
plot   
    


# # Train

# In[ ]:


#Fare
df['Fare_Cat']=0
df.loc[df['Fare']<=7.01,'Fare_Cat']=1
df.loc[(df['Fare']>7.01) & (df['Fare']<=7.88),'Fare_Cat']=2
df.loc[(df['Fare']>7.88) & (df['Fare']<=69.4),'Fare_Cat']=3
df.loc[(df['Fare']>69.4) & (df['Fare']<=77),'Fare_Cat']=4
df.loc[df['Fare']>77,'Fare_Cat']=5


# In[ ]:


fig = plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm',linewidths=0.2)
plt.title('Matriz de correlaciones', fontsize=22)
plt.show()


# In[ ]:


df.head()


# In[ ]:


Variables=['Embarked','Cabin','Title','Pclass','Ticket']
for label in Variables:
    df1=pd.get_dummies(df[label],prefix=label)
    df1=df.join(df1)
    df=df1   


# In[ ]:


df=df.drop(Variables,axis=1)


# In[ ]:


df.head()


# In[ ]:


delete_train=['Name', 'SibSp','Parch','Fare']
df=df.drop(delete_train,axis=1)


# In[ ]:


df.head()


# In[ ]:


# calculate the correlation matrix (ignore survived and passenger id fields)
df_corr = df.drop(['PassengerId','Survived'],axis=1).corr(method='spearman')

# create a mask to ignore self-
mask = np.ones(df_corr.columns.size) - np.eye(df_corr.columns.size)
df_corr = mask * df_corr

drops = []
# loop through each variable
for col in df_corr.columns.values:
    # if we've already determined to drop the current variable, continue
    if np.in1d([col],drops):
        continue
    
    # find all the variables that are highly correlated with the current variable 
    # and add them to the drop list 
    corr = df_corr[abs(df_corr[col]) > 0.7].index
    drops = np.union1d(drops, corr)

print ("nDropping", drops.shape[0], "highly correlated features...n", drops)
df.drop(drops, axis=1, inplace=True)


# In[ ]:


train=df.loc[df.Survived.notnull()].reset_index(drop=True)
test=df.loc[df.Survived.isnull()].drop('Survived',axis=1).reset_index(drop=True)

X=train.drop(['PassengerId','Survived'], axis=1)
y=train['Survived']


# In[ ]:


from sklearn.preprocessing import StandardScaler
std=StandardScaler()

X_scale=std.fit_transform(X)
X_test=std.transform(test.drop('PassengerId',axis=1))

X_scale=pd.DataFrame(X_scale,columns=X.columns.tolist())
X_test=pd.DataFrame(X_test,columns=X.columns.tolist())


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve

x_train, x_test, y_train, y_test = train_test_split(X_scale,y, test_size=0.3,stratify=y)



# # Selección de variables

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(random_state=123,n_estimators=50, max_features='sqrt')
rf.fit(x_train,y_train)



# In[ ]:


features_importances=pd.DataFrame(rf.feature_importances_.tolist())
features_labels=pd.DataFrame(x_train.columns.tolist())
df_features=pd.concat([features_labels,features_importances],axis=1)
df_features.columns =['Var','Importance']


# In[ ]:


fig, axis =plt.subplots()

df_f=df_features.sort_values(by='Importance',ascending=False).reset_index(drop=True)
#condition=df_f.loc[df_f['Importance']>=0.01].sort_values(by='Importance',ascending=True)
condition=df_f.sort_values(by='Importance',ascending=True)
condition['%']=np.round(condition["Importance"]/sum(condition['Importance'])*100,2)
index=np.arange(condition.shape[0])
bar_width = 0.5
rects = plt.barh(index , condition["Importance"]/sum(condition['Importance']), bar_width, alpha=0.4, color='g', label='Main')
plt.title('Variables más influyentes',fontsize=30,color='black')
plt.yticks(index,condition['Var'],fontsize=15)
fig.set_size_inches(20.5, 20.5)
plt.show()


# In[ ]:


from sklearn.feature_selection import SelectFromModel

model = SelectFromModel(rf, prefit=True)
x_train_reduced = model.transform(x_train)
x_test = model.transform(x_test)
feature_idx = model.get_support()
feature_name = x_train.columns[feature_idx]







# In[ ]:


x_test=pd.DataFrame(x_test,columns=feature_name)


# In[ ]:


#Balanceo


# In[ ]:


from imblearn.combine import SMOTETomek
smt = SMOTETomek(ratio='auto')
x_smt, y_smt = smt.fit_sample(x_train_reduced,y_train)



# In[ ]:


x_smt=pd.DataFrame(x_smt,columns=feature_name)
y_smt=pd.DataFrame(y_smt)
y_smt.columns=['y']


# In[ ]:


#Save results
Model=[]
fit=[]
Accuracy=[]
Precision=[]
Recall=[]



# # Model Regresión Logística

# In[ ]:


from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(x_smt,y_smt)


# In[ ]:


y_test_p=lg.predict(x_test)


# In[ ]:


print('El Accuracy es: %0.2f'%accuracy_score(y_test,y_test_p))
print('La Precisión es: %0.2f'%precision_score(y_test,y_test_p))
print('El Recall es: %0.2f'%recall_score(y_test,y_test_p))


# In[ ]:


Model.append('LogisticRegression')
fit.append('lg')
Accuracy.append(np.round(accuracy_score(y_test,y_test_p),4))
Precision.append(np.round(precision_score(y_test,y_test_p),4))
Recall.append(np.round(recall_score(y_test,y_test_p),4))


# # Model Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_smt,y_smt)


# In[ ]:


y_test_p=rf.predict(x_test)


# In[ ]:


print('El Accuracy es: %0.2f'%accuracy_score(y_test,y_test_p))
print('La Precisión es: %0.2f'%precision_score(y_test,y_test_p))
print('El Recall es: %0.2f'%recall_score(y_test,y_test_p))


# In[ ]:


Model.append('RandomForestClassifier')
fit.append('rf')
Accuracy.append(np.round(accuracy_score(y_test,y_test_p),4))
Precision.append(np.round(precision_score(y_test,y_test_p),4))
Recall.append(np.round(recall_score(y_test,y_test_p),4))


# # Model Gradient Boost

# In[ ]:


# create param grid object 
forrest_params = dict(     
    max_depth = [n for n in range(9, 14)],     
    min_samples_split = [n for n in range(4, 11)], 
    min_samples_leaf = [n for n in range(2, 5)],     
    n_estimators = [n for n in range(10, 60, 10)],
)


# In[ ]:


# instantiate Random Forest model
from sklearn.model_selection import GridSearchCV
forrest = RandomForestClassifier()


# In[ ]:


# build and fit model 
forest_cv = GridSearchCV(estimator=forrest,     param_grid=forrest_params, cv=5) 

forest_cv.fit(x_smt,y_smt)


# In[ ]:


y_test_p=forest_cv.predict(x_test)


# In[ ]:


print('El Accuracy es: %0.2f'%accuracy_score(y_test,y_test_p))
print('La Precisión es: %0.2f'%precision_score(y_test,y_test_p))
print('El Recall es: %0.2f'%recall_score(y_test,y_test_p))


# In[ ]:


Model.append('GridSearchCV')
fit.append('forest_cv')
Accuracy.append(np.round(accuracy_score(y_test,y_test_p),4))
Precision.append(np.round(precision_score(y_test,y_test_p),4))
Recall.append(np.round(recall_score(y_test,y_test_p),4))


# # Modelo XGBoost

# In[ ]:


import xgboost as xgb
XGBoost=xgb.XGBClassifier(n_jobs=-1,n_estimators=300,subsample=0.75,max_depth=10)
XGBoost.fit(x_smt, y_smt)


# In[ ]:


y_test_p=XGBoost.predict(x_test)


# In[ ]:


print('El Accuracy es: %0.2f'%accuracy_score(y_test,y_test_p))
print('La Precisión es: %0.2f'%precision_score(y_test,y_test_p))
print('El Recall es: %0.2f'%recall_score(y_test,y_test_p))


# In[ ]:


Model.append('XGBoost')
fit.append('XGBoost')
Accuracy.append(np.round(accuracy_score(y_test,y_test_p),4))
Precision.append(np.round(precision_score(y_test,y_test_p),4))
Recall.append(np.round(recall_score(y_test,y_test_p),4))


# # Model Adaboost

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
# Parámetros :
# base_estimator : Es el estimador base sobre la cual el ensamble es constuido.
# n_estimators : Numero de estimadores con los cuales se construye el ensamble.
# random_state : semilla aleatoria
AdaBoost=AdaBoostClassifier(learning_rate=0.01, n_estimators=100)
AdaBoost.fit(x_smt, y_smt) 


# In[ ]:


y_test_p=AdaBoost.predict(x_test)


# In[ ]:


print('El Accuracy es: %0.2f'%accuracy_score(y_test,y_test_p))
print('La Precisión es: %0.2f'%precision_score(y_test,y_test_p))
print('El Recall es: %0.2f'%recall_score(y_test,y_test_p))


# In[ ]:


Model.append('AdaBoost')
fit.append('AdaBoost')
Accuracy.append(np.round(accuracy_score(y_test,y_test_p),4))
Precision.append(np.round(precision_score(y_test,y_test_p),4))
Recall.append(np.round(recall_score(y_test,y_test_p),4))


# In[ ]:


results=pd.DataFrame({'Model':Model,'fit':fit,'accuracy':Accuracy,'precision':Precision,'recall':Recall})


# In[ ]:


results.sort_values(by='accuracy',ascending=False)


# In[ ]:


# dataframe with predictions

passengerId=test.reset_index()['PassengerId']
forrest_pred = lg.predict(X_test[feature_name])
# dataframe with predictions
my_submission = pd.DataFrame({'PassengerId': passengerId, 'Survived': forrest_pred.astype('int64')})
# save to csv
my_submission.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:




