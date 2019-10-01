#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import the Libraries
#Import a few libraries you think you'll need (Or just import them as you go along!)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
get_ipython().magic(u'matplotlib inline')


# In[ ]:


#Get the Data
#Read in the train.csv,test.csv file and set it to a pandas data frame called df_train and df_test.
df_train =pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")


# In[ ]:


#Check the head of df_train
df_train.head()


# In[ ]:


#Check how many rows and columns are there
df_train.info()


# In[ ]:


#Define the importnant behaviours using describe function
df_train.describe()


# In[ ]:


#We can use seaborn to create a simple heatmap to see where we are missing data!
sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


#Of all passengers in df_train, how many survived, how many died ?
sns.countplot(x='Survived', data=df_train);


# In[ ]:


#More people died than survived (38% survived)
print(df_train.Survived.sum()/df_train.Survived.count())


# In[ ]:


#Female more likely to survive than male
df_train.groupby(['Survived','Sex'])['Survived'].count()


# In[ ]:


sns.factorplot(x='Sex', col='Survived', kind='count', data=df_train);


# In[ ]:


print("% of women survived: " , df_train[df_train.Sex == 'female'].Survived.sum()/df_train[df_train.Sex == 'female'].Survived.count())
print("% of men survived:   " , df_train[df_train.Sex == 'male'].Survived.sum()/df_train[df_train.Sex == 'male'].Survived.count())


# In[ ]:


pd.crosstab(df_train.Pclass, df_train.Survived, margins=True).style.background_gradient(cmap='autumn_r')


# In[ ]:


print("% of survivals in") 
print("Pclass=1 : ", df_train.Survived[df_train.Pclass == 1].sum()/df_train[df_train.Pclass == 1].Survived.count())
print("Pclass=2 : ", df_train.Survived[df_train.Pclass == 2].sum()/df_train[df_train.Pclass == 2].Survived.count())
print("Pclass=3 : ", df_train.Survived[df_train.Pclass == 3].sum()/df_train[df_train.Pclass == 3].Survived.count())


# In[ ]:


#Get the Survived data from Pclass
sns.factorplot('Pclass','Survived', data=df_train)
plt.show()


# In[ ]:


pd.crosstab([df_train.Sex, df_train.Survived], df_train.Pclass, margins=True).style.background_gradient(cmap='autumn_r')


# In[ ]:


#Get the Survived data from Pclass and Sex
sns.factorplot('Pclass','Survived',hue='Sex',data=df_train)
plt.show()


# In[ ]:


sns.factorplot(x='Survived', col='Embarked', kind='count', data=df_train);


# In[ ]:


sns.factorplot('Embarked','Survived', data=df_train)
plt.show()


# In[ ]:


sns.factorplot('Embarked','Survived', hue= 'Sex', data=df_train)
plt.show()


# In[ ]:


sns.factorplot(x='Survived', col='SibSp', kind='count', data=df_train);


# In[ ]:


print("% of sibbling 0 survived: " , df_train[df_train.SibSp == 0].Survived.sum()/df_train[df_train.SibSp == 0].Survived.count())
print("% of sibbling 1 survived:   " , df_train[df_train.SibSp == 1].Survived.sum()/df_train[df_train.SibSp == 1].Survived.count())
print("% of sibbling 2 survived: " , df_train[df_train.SibSp == 2].Survived.sum()/df_train[df_train.SibSp == 2].Survived.count())
print("% of sibbling 3 survived:   " , df_train[df_train.SibSp == 3].Survived.sum()/df_train[df_train.SibSp == 3].Survived.count())
print("% of sibbling 4 survived: " , df_train[df_train.SibSp == 4].Survived.sum()/df_train[df_train.SibSp == 4].Survived.count())
print("% of sibbling 5 survived:   " , df_train[df_train.SibSp == 5].Survived.sum()/df_train[df_train.SibSp == 5].Survived.count())
print("% of sibbling 6 survived:   " , df_train[df_train.SibSp == 6].Survived.sum()/df_train[df_train.SibSp == 6].Survived.count())


# In[ ]:


sns.factorplot(x='Survived', col='Parch', kind='count', data=df_train);


# In[ ]:


print("% of Parch 0 survived: " , df_train[df_train.Parch == 0].Survived.sum()/df_train[df_train.Parch == 0].Survived.count())
print("% of Parch 1 survived:   " , df_train[df_train.Parch == 1].Survived.sum()/df_train[df_train.Parch == 1].Survived.count())
print("% of Parch 2 survived: " , df_train[df_train.Parch == 2].Survived.sum()/df_train[df_train.Parch == 2].Survived.count())
print("% of Parch 3 survived:   " , df_train[df_train.Parch == 3].Survived.sum()/df_train[df_train.Parch == 3].Survived.count())
print("% of Parch 4 survived: " , df_train[df_train.Parch == 4].Survived.sum()/df_train[df_train.Parch == 4].Survived.count())
print("% of Parch 5 survived:   " , df_train[df_train.Parch == 5].Survived.sum()/df_train[df_train.Parch == 5].Survived.count())
print("% of Parch 6 survived:   " , df_train[df_train.Parch == 6].Survived.sum()/df_train[df_train.Parch == 6].Survived.count())


# In[ ]:


pd.crosstab([df_train.Sex, df_train.Survived], [df_train.SibSp, df_train.Pclass], margins=True).style.background_gradient(cmap='autumn_r')


# In[ ]:


sns.distplot(df_train['Fare'])
plt.show()


# In[ ]:


pd.crosstab([df_train.Fare,  df_train.Pclass], df_train.Survived, margins=True).style.background_gradient(cmap='autumn_r')


# In[ ]:


pd.crosstab([df_train.Survived], [df_train.Sex, df_train.Pclass, df_train.Embarked, df_train.Age, df_train.SibSp, df_train.Parch, df_train.Fare], margins=True)


# In[ ]:


#Data Preprocessing
#Clone two new dataframes df_train_ml and df_test_ml to avoid any alters
df_train_ml = df_train.copy()
df_test_ml = df_test.copy()


# In[ ]:


#Get the mean value of the 'Age' and set to the 'nan' values
df_train_ml['Age'].fillna(value=df_train_ml['Age'].mean(), inplace=True)


# In[ ]:


#Converting categorical to numerical by pd.get_dummies
df_train_ml = pd.get_dummies(df_train_ml, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)
df_train_ml.drop(['PassengerId','Name','Ticket', 'Cabin'],axis=1,inplace=True)


# In[ ]:


df_train_ml.info()


# In[ ]:


df_test_ml = pd.get_dummies(df_test_ml, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)
df_test_ml.drop(['PassengerId','Name','Ticket', 'Cabin'],axis=1,inplace=True)


# In[ ]:


df_test_ml.info()


# In[ ]:


corr = df_train_ml.corr()
sns.heatmap(corr)
plt.show()


# In[ ]:


#Select Scaler and transform the data and add it to the Data Frame.
scaler = StandardScaler()

scaler.fit(df_train_ml.drop('Survived',axis=1))
scaled_features = scaler.transform(df_train_ml.drop('Survived',axis=1))
df_train_ml_sc = pd.DataFrame(scaled_features, columns=df_train_ml.columns[[1,2,3,4,5,6,7,8,9]]) 

df_train_ml_sc.info()


# In[ ]:


df_test_ml.fillna(df_test_ml.mean(), inplace=True)
scaler.fit(df_test_ml)
scaled_features = scaler.transform(df_test_ml)
df_test_ml_sc = pd.DataFrame(scaled_features, columns=df_test_ml.columns)

df_test_ml_sc.info()


# In[ ]:


#Split the train dataset into two categories. 75% for Train data and 25% for Test data. Disable data shuffling to get the best accuracy
X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(df_train_ml_sc, df_train_ml['Survived'], test_size=0.25,shuffle=False)
#X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(df_train_ml_sc, df_train_ml['Survived'], test_size=0.30,random_state=101)


# In[ ]:


# scaled data assigned to variables
X_train_all_sc = X_train_sc
y_train_all_sc = y_train_sc
X_test_all_sc = X_test_sc

X_test_data = df_test_ml_sc


# In[ ]:


#put the scaled data into arrays
X = X_train_all_sc
X = np.array(X)
y = y_train_all_sc
y = np.array(y)
X_test_predict = np.array(X_test_all_sc)
Test_data = np.array(X_test_data)


# In[ ]:


#select the necessary moel for train the data
modele = GaussianNB()

# Train the model using the training sets 
modele.fit(X, y)

#Predict the model
pred_modele = modele.predict(X_test_predict)

#Accuracy level gain from prediction model and y_test_sc data.
print(confusion_matrix(y_test_sc, pred_modele))
print(classification_report(y_test_sc, pred_modele))
print(accuracy_score(y_test_sc, pred_modele))


# In[ ]:


#Get the output of the given test data
predicted = modele.predict(Test_data)
print(predicted)


# In[ ]:


df_test_final = df_test.copy()
df_test_final['Survived'] = predicted


# In[ ]:


pd.crosstab([df_test_final.Sex, df_test_final.Survived,  df_test_final.Age], [df_test_final.SibSp, df_test_final.Pclass], margins=True).style.background_gradient(cmap='autumn_r')


# In[ ]:





# In[ ]:




