#!/usr/bin/env python
# coding: utf-8

# I did basic EDA of all the features available in the dataset so as to analyze how the features are related with survival rate. Further after EDA some features are removed while some features are also added to the dataset.
# Finally I have done prediction using some machine learning algorithms like Knearest, Decision tree, Random Forest and Logistic Regression.
# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
get_ipython().magic(u'matplotlib inline')


# In[ ]:


data = pd.read_csv('../input/train.csv')


# In[ ]:


data.head()


# In[ ]:


data.isnull()


# In[ ]:


data.isnull().sum()


# GENDER : How Gender is related with the survival rate.

# In[ ]:


pd.crosstab(data.Sex,data.Survived,margins=True).style.background_gradient(cmap='summer_r')


# In[ ]:


data.groupby(['Sex','Survived'])['Survived'].count()


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(10,5))
data['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.2f%%',ax=ax[0])
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived', data=data, ax=ax[1])
ax[1].set_title('Survived')
plt.show()


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(10,5))
data[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex', hue='Survived', data=data, ax=ax[1])
ax[1].set_title('Survivied vs Dead')
plt.show()


# Inference from Gender:
# 1.Of the total passengers on the ship 65% were male passengers.
# 2.We see that priority to survive is given to females as compared to males. Out of the total females present 75% survived, whereas only about 20% males survived out of total 577 males.

# Passenger Class relation with survival:

# In[ ]:


data.groupby(['Pclass', 'Survived'])['Survived'].count()


# In[ ]:


pd.crosstab(data.Pclass,data.Survived,margins=True).style.background_gradient(cmap='Wistia')


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(10,5))
data['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])
ax[0].set_title('Passengers by Pclass')
ax[0].set_ylabel('Count')
sns.countplot('Pclass',hue='Survived',data=data,ax=ax[1])
ax[1].set_title('Pclass-Survived vs Dead')
plt.show()

Inference from Pclass :
1.Class 3 has max no of passengers. ie 55% are in class 3.
2.More no of passenger are saved from class 1 percentage wise (62%).
3.No other class has survival rate of more than 50%.
4.The class that is costlier has more survival rate. Money speaks!
# Pclass and Sex Relation

# In[ ]:


pd.crosstab([data.Sex,data.Survived],data.Pclass,margins=True).style.background_gradient(cmap='cool')


# In[ ]:


sns.factorplot('Pclass','Survived', hue = 'Sex', data=data)
plt.show()


# Inference for Gender and Pclass Relation:
# 1.Almost all females of class-1 survived barring 3 out of 94 and 70 out of 76 survived from the class 2 and 50% survived from class3.
# 2.So the maximum priority was given to class-1 and class-2 female passenger their survival rate was 97% and 92% respectively.
# 3.Incase of males 37% is the survival rate for class 1. For class 2 and 3 survival rate is pretty low.

# Age Relation for Survival :

# In[ ]:


print('Oldest Passenger was ',data['Age'].max(), 'years old.' )
print('Youngest Passenger was ',data['Age'].min(), 'years old')
print('Average age for the passengers was ', data['Age'].mean(), 'Years')


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(10,5))
sns.violinplot('Pclass', 'Age', hue = 'Survived', data=data, split=True, ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))
sns.violinplot('Sex', 'Age', hue = 'Survived', data=data, split=True, ax=ax[1])
ax[1].set_title('Age and Sex vs Survival')
ax[1].set_yticks(range(0,110,10))
plt.show()


# Inference from Age Sex and Pclass with Survival:
# 1.From class 1 more passengers between 10-50 years of age survived, from class 2 passengers between 20-40 years of age survived more and in class 3 the survival rate was more for age 10-35years.
# 2.More priority was given to young passengers. Very few passenger of age 0 to 15 did not survive.
# 3.Considering females the survival rate was more for age 10-50.
# 4.Many males in the age group 18-45 were unfortunate.
# 5.Overall Survival rate for children is good.

# In[ ]:


data['Initial']=0
for i in data:
    data['Initial']=data.Name.str.extract('([A-Za-z]+)\.')    


# In[ ]:


pd.crosstab(data.Initial,data.Sex).T.style.background_gradient(cmap='hsv')


# In[ ]:


data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)


# In[ ]:


data.groupby('Initial')['Age'].mean()


# In[ ]:


data.loc[(data.Age.isnull())&(data.Initial=='Mr'),'Age']=33
data.loc[(data.Age.isnull())&(data.Initial=='Mrs'),'Age']=36
data.loc[(data.Age.isnull())&(data.Initial=='Master'),'Age']=5
data.loc[(data.Age.isnull())&(data.Initial=='Miss'),'Age']=22
data.loc[(data.Age.isnull())&(data.Initial=='Others'),'Age']=46


# In[ ]:


data.isnull().sum()


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(15,10))
data[data['Survived']==0].Age.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')
ax[0].set_title('Survived=0')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
data[data['Survived']==1].Age.plot.hist(ax=ax[1],bins=20,edgecolor='black',color='green')
ax[1].set_title('Survived=1')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)
plt.show()


# Inference:
# 1.Max children were saved ie 0-5yrs.
# 2.Very few senior citizens survived the disaster nobody from 65-75 yrs of age survived.
# 3.The oldest passengers were saved.
# 

# In[ ]:


sns.factorplot('Pclass','Survived',col='Initial',data=data)
plt.show()


# Inference:
# 1.From the above figs we note Children and women from class 1 were given the maximum priority for survival followed by thye same from class 2 and then 3.
# 2.Men from class 3 were given least priority for survival.
# 

# Survival chances based on port of Embarkation for Pclass and Gender.

# In[ ]:


pd.crosstab([data.Embarked,data.Pclass],[data.Sex,data.Survived],margins=True).style.background_gradient(cmap='autumn')


# In[ ]:


sns.factorplot('Embarked', 'Survived', data=data)
fig=plt.gcf()
fig.set_size_inches(5,3)
plt.show()


# Inference:
# 1.Survival rate is very much high for class 1 female passenger that embarked from Cherbourg. Only 1 female from this category didnt survive the disaster.
# 2.The survival rate for males is also good (>60%) for Cherbourg port.
# 3.There were very few people who embarked from Queensland. Nobody survived from class 1 and class 2 men whereas no female died from class 1 and class 2 who embarked from Queensland.
# 4.The class 3 passenger of both ports Queensland and Southampton did not have a good survival rate.
# 5.Females from class 1 and 2 of Southampton had a good survival rate whereas men did not.
# 6.Port Cherbourg has the highest survival chances.
# 
#     

# In[ ]:


f,ax=plt.subplots(2,2,figsize=(15,10))
sns.countplot('Embarked',data=data,ax=ax[0,0])
ax[0,0].set_title('Passengers Boarded')
sns.countplot('Embarked',hue='Sex',data=data,ax=ax[0,1])
ax[0,1].set_title('Gender Embarked')
sns.countplot('Embarked',hue='Survived',data=data,ax=ax[1,0])
ax[1,0].set_title('Embarked vs Survived')
sns.countplot('Embarked',hue='Pclass',data=data,ax=ax[1,1])
ax[1,1].set_title('Embarked vs Pclass')
plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.show()


# In[ ]:


sns.factorplot('Pclass','Survived', hue='Sex',col='Embarked',data=data)
plt.show()


# Inference:
# 1.More no of people boarded from Southapmton.
# 2.Cherbourg had more no of survivors than the victims ie more than 50% of Cherbourg passengers survived.
# 3.Cherbourg had more no of class 1 passengers which led to more survivor rate for Cherbourg passengers.
# 4.Queensland had least class 1 or class 2 passengers.
# 5.Females from all 3 ports class 1 and 2 had a good survival rate and those from class 3 of Queensland also had higher survival rate.
# 6.Class 3 of Southampton does'nt have a good survival rate either for men or women.

# Dealing with Null values of Embarked:
# Use S as default for null values as max passengers boarded from S.

# In[ ]:


data['Embarked'].fillna('S', inplace=True)


# In[ ]:


data.Embarked.isnull().any()


# Realtion for siblings or spouse

# In[ ]:


pd.crosstab([data.SibSp],data.Survived,margins=True).style.background_gradient(cmap='summer_r')


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(15,5))
sns.barplot('SibSp','Survived',data=data,ax=ax[0])
ax[0].set_title('Siblings vs Survived')
sns.factorplot('SibSp','Survived',data=data,ax=ax[1])
ax[1].set_title('SibSp vs Survived')
plt.close(2)
plt.show()


# In[ ]:


pd.crosstab(data.SibSp,data.Pclass).style.background_gradient(cmap='summer_r')


# Inference:
# 1.Most Passengers did not have any sibling.
# 2.We can see passengers having 1 or 2 siblings/spouse had high chances of survival about 50%.
# 3.Passengers having 3 or more siblings/spouse had very less chances of survival.
# 4.Anybody having 1 sibling/spouse had good chances of survival >50%.
# 5.Almost all passengers having 3 or more siblings belonged to class 3, while those with 1 siblings were equally distributed in all 3 classes.

# Parent Child Relation:

# In[ ]:


pd.crosstab(data.Parch,data.Pclass).style.background_gradient(cmap='summer_r')


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(15,5))
sns.barplot('Parch','Survived',data=data,ax=ax[0])
ax[0].set_title('Parch vs Survived')
sns.factorplot('Parch','Survived',data=data,ax=ax[1])
ax[1].set_title('Parch vs Survived')
plt.close(2)
plt.show()


# Inference:
# 1.Many passengers who did not have parents or children belonged to class 1.
# 2.Passenger with 1-3 parents/children had a good chance for survival 50-60%.
# 3.Passengers with 4-5 children have a poor survival rate.
# 4.Children had a good chance for survival with parents on board.

# Fare Relation

# In[ ]:


print('Highest Fare was:',data['Fare'].max())
print('Lowest Fare was:',data['Fare'].min())
print('Average Fare was:',data['Fare'].mean())


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(15,10))
data[data['Survived']==0].Fare.plot.hist(ax=ax[0],bins=30,edgecolor='black',color='red')
ax[0].set_title('Survived= 0')
x1=list(range(0,550,50))
ax[0].set_xticks(x1)
data[data['Survived']==1].Fare.plot.hist(ax=ax[1],color='green',bins=30,edgecolor='black')
ax[1].set_title('Survived= 1')
x2=list(range(0,550,50))
ax[1].set_xticks(x2)
plt.show()


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(10,5))
sns.violinplot("Pclass","Fare", hue="Survived", data=data,split=True,ax=ax[0])
ax[0].set_title('Pclass and Fare vs Survived')
ax[0].set_yticks(range(0,550,50))
sns.violinplot("Sex","Fare", hue="Survived", data=data,split=True,ax=ax[1])
ax[1].set_title('Sex and Fare vs Survived')
ax[1].set_yticks(range(0,550,50))
plt.show()


# Inference:
# 1.Survival rate doenst depend much on fare.
# 2.The costliest fare passenger survived.

# Inference Summary:
# 
# Sex: Females have a greater chances of survival than men.
# Pclass: Class 1 passengers have more probability of survival than 2 followed by 3. So the costlier the price more is the chance for survival.
# Age: Children of age group 0-15 have higher chances for survival. Many passengers between 15-35 were unfortunate.
# Embarked: The passengers who embarked from Cherbourg had a good chance for survival than Southampton and Queensland.
# Parch and SibSp: Passengers having 1-3 parents/children had good chances for survival than others and passengers with 1 or 2 siblings also had more than 50% chances for survival.
# 

# In[ ]:


sns.heatmap(data.corr(),annot=True,cmap='PiYG',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()


# Feature Engineering & Data Cleaning:
# All the feautures present in the table are not necessary for prediction of data.
# We can eliminate unwanted features or even add new features for accurate data prediction.
# We will analyze all the features:
# 
# AGE: It is a continuous feature. Machine learning models dont support continuous features well.
# To categorize people into different groups is difficult as age is a continuous factor.
# We can categorize age in sections. Each section has a age gap. The total age varies from 0-80yrs, so we can divide it into 5 sections ie o-16, 17-32 and so on. Hence we can eliminate the Age feature.
# 

# In[ ]:


data['Age_band']=0
data.loc[data['Age']<=16,'Age_band']=0
data.loc[(data['Age']>16)&(data['Age']<=32),'Age_band']=1
data.loc[(data['Age']>32)&(data['Age']<=48),'Age_band']=2
data.loc[(data['Age']>48)&(data['Age']<=64),'Age_band']=3
data.loc[(data['Age']>64)&(data['Age']<=80),'Age_band']=4
data.head(2)


# In[ ]:


data['Age_band'].value_counts().to_frame()


# Most passengers were from the age group 17 to 32. 
# Very few passenger were above 65.

# In[ ]:


sns.factorplot('Age_band','Survived',data=data,col='Pclass')
plt.show()


# In[ ]:


sns.barplot('Age_band','Survived',data=data)


# Inference: Survival rate decreases with age irresective of the class. Class 4 has low survival rate. 

# In[ ]:


data['Family']=0
data['Family']=data['Parch']+data['SibSp']
data['Alone']=0
data.loc[data.Family==0,'Alone']=1
data.head(3)


# We have combined siblings/spouse and parent/children into a new feature called family, whereas those who dont have any family are put into feature alone. Hence we can eliminate SibSp and Parch.

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(10,5))
sns.factorplot('Family','Survived',data=data,ax=ax[0])
ax[0].set_title('Family vs Survived')
sns.factorplot('Alone','Survived',data=data,ax=ax[1])
ax[1].set_title('Alone vs Survived')
plt.close(2)
plt.close(3)
plt.show()


# Family is a good parameter for survival. As the family members increases (>4) survival rate decreases. Alone people have less survival chances than family people. for eg In a grouping charting if a single person is saved in a family, then the probability of survival goes up for family.

# In[ ]:


sns.factorplot('Alone','Survived',data=data,hue='Sex',col='Pclass')


# 1=Alone 0=Family. 
# Travelling alone has less survival rates than going with family irrespective of Pclass and Gender.

# In[ ]:


data['Fare_range']=pd.qcut(data['Fare'],5)
data.groupby(['Fare_range'])['Survived'].mean().to_frame().style.background_gradient(cmap='summer_r')


# Like age, fare is also a continuous feature. So we have grouped age into 5 sections.
# The costlier the fare more are chances for surviving. We will give single values to Fare_range.
# 

# In[ ]:


data['Fare_cat']=0
data.loc[data['Fare']<=0.7854,'Fare_cat']=0
data.loc[(data['Fare']>0.7854)&(data['Fare']<=10.5),'Fare_cat']=1
data.loc[(data['Fare']>10.5)&(data['Fare']<=21.679),'Fare_cat']=2
data.loc[(data['Fare']>21.679)&(data['Fare']<=39.688),'Fare_cat']=3
data.loc[(data['Fare']>39.688)&(data['Fare']<=512.329),'Fare_cat']=4


# In[ ]:


sns.factorplot('Fare_cat','Survived',data=data,hue='Sex')
plt.show()


# Fare_cat is an important feature 
# Now lets convert everything into numerical values.
# 

# In[ ]:


data['Sex'].replace(['male','female'],[0,1],inplace=True)
data['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
data['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)


# Now we can clean/remove unwanted data
# Name-We have the initals of the name so we can eliminate name feature.
# Age-As it a continuous feature and we have already split it into Age_band we can eliminate age feature.
# Ticket-It consist of random strings.
# Fare-It is also a continuous feature and is already split into Fare_cat.
# Cabin-It has 90% null values.
# Fare range-It is converted into single values Fare_cat.
# PassengerId-It has nothing to do with survival rate.
# 

# In[ ]:


data.head(2)


# In[ ]:


data.drop(['Name','Age','Ticket','Fare','Cabin','Fare_range','PassengerId'],axis=1,inplace=True)
sns.heatmap(data.corr(),annot=True,cmap='PiYG',linewidths=0.2,annot_kws={'size':8})
fig=plt.gcf()
fig.set_size_inches(8,6)
plt.show()


# In[ ]:


data.head(2)


# PREDICTIVE MODELLING:
# 

# In[ ]:


from sklearn.linear_model import LogisticRegression 
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.cross_validation import train_test_split 
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix


# In[ ]:


train,test=train_test_split(data,test_size=0.3,random_state=0,stratify=data['Survived'])
train_X=train[train.columns[1:]]
train_Y=train[train.columns[:1]]
test_X=test[test.columns[1:]]
test_Y=test[test.columns[:1]]
X=data[data.columns[1:]]
Y=data['Survived']


# Logistic Regression:

# In[ ]:


model = LogisticRegression()
model.fit(train_X,train_Y)
prediction1=model.predict(test_X)
print('The accuracy is for Logistic Regression is ',metrics.accuracy_score(prediction1,test_Y))


# In[ ]:


model=DecisionTreeClassifier()
model.fit(train_X,train_Y)
prediction2=model.predict(test_X)
print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction2,test_Y))


# In[ ]:


model=RandomForestClassifier()
model.fit(train_X,train_Y)
prediction3=model.predict(test_X)
print('The accuracy of the Random Forests is',metrics.accuracy_score(prediction3,test_Y))


# In[ ]:


model=KNeighborsClassifier() 
model.fit(train_X,train_Y)
prediction4=model.predict(test_X)
print('The accuracy of the KNN is',metrics.accuracy_score(prediction4,test_Y))


# Confusion Matrix
# 

# In[ ]:


from sklearn.model_selection import cross_val_predict 


# In[ ]:


f,ax=plt.subplots(2,2,figsize=(10,8))
y_pred = cross_val_predict(RandomForestClassifier(n_estimators=100),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,0],annot=True,fmt='2.0f')
ax[0,0].set_title('Matrix for Random-Forests')
y_pred = cross_val_predict(LogisticRegression(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[0,1],annot=True,fmt='2.0f')
ax[0,1].set_title('Matrix for Logistic Regression')
y_pred = cross_val_predict(DecisionTreeClassifier(),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,0],annot=True,fmt='2.0f')
ax[1,0].set_title('Matrix for Decision Tree')
y_pred = cross_val_predict(KNeighborsClassifier(n_neighbors=9),X,Y,cv=10)
sns.heatmap(confusion_matrix(Y,y_pred),ax=ax[1,1],annot=True,fmt='2.0f')
ax[1,1].set_title('Matrix for KNN')
plt.show()



# Left diagonal matrix are correct prediction while right diagonal are wrong predictions.
# Decision Tree predicts best for dead predictions.
# K nearest predicts best for survivals.
# K nearest classification has the best predcition overall. 
