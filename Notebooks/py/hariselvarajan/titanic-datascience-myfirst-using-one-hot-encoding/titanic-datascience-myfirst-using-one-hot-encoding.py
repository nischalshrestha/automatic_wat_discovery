#!/usr/bin/env python
# coding: utf-8

# OK!!! So I have started a new journey into the world of Data Science, Machine Learning and AI. What better place to start than in Kaggle :-)
#     
# So here's my attempt on **Titanic Data Set problem**
# 

# In[ ]:



#First lets import the required libraries
import pandas as pd


# import visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

# import machine learning libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# After having worked on over 50 version of my code, i realized data knowledge on the concepts of machine learning alone will not be enough. You also need know have a strong hold on the concepts of data science.
# Many of the submission made, the key thing which makes a submission get higher score is not much to do with the model used, but on how the data is **enriched**.
# 
# 

# In[ ]:



# First lets combine the train and test data set as we would be doing few clean up task and perform some feature engineering.
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
allset=train_df.append(test_df, sort=False)

#Data preview
allset.head(20)


# Lets have a look at the following 3 features
# * Sex
# * Pclass
# * Embarked
# 
# A picture can explain better than words...

# In[ ]:


# The output variable to predict is survived. We will try to check how the various feature provided affect the output variable
# Plotting a graph with Sex on x-axis and average survival rate on Y-axis below:
Chart, items =plt.subplots(1,3,figsize=(25,5))
CGender = sns.barplot(x="Sex",y="Survived",data=allset,ax=items[0])
CGender = CGender.set_ylabel("Survival Probability")
CClass = sns.barplot(x="Pclass",y="Survived",data=allset,ax=items[1])
CClass = CClass.set_ylabel("Survival Probability")
CEmbarked = sns.barplot(x="Embarked",y="Survived",data=allset,ax=items[2])
CEmbarked = CEmbarked.set_ylabel("Survival Probability")


# As you can see from the graph above, all 3 features have a good correlation with teh output variable Survived. 
#  - female have higher chance of survival than males
#  - passeger in the higher class (class 1) have more chances of survival than in lower classes
#  - passenger who have boarded from point C have higher survival rate, and passenger who have boarded from S, have the lowest survival rate
#   
#   Considering the above, we will include all these 3 features as input to our model( after one key change..shown later)
#   
#   Let's Look at Age Parameter. Some work to be done here.
#   First of all age field is a numerical field. In Data science world, its also called continuous variable. In order to its importance, its best to convert it into a categorical variable.
#   We will group age into different buckets. But before that, some cleaning up to be done. Age feature has about 263 values which are null. We would need to fill this up with our best guess possible.

# In[ ]:


allset.Age.isnull().sum()


# We could update these blank values with the average age, but updating about 20% of the data with a single avg value might add more noise to the data set. Lets try to look how the age is related to other features
# 

# In[ ]:


AgeGroup=allset[['Age', 'Pclass','Sex']].groupby(['Sex','Pclass'], as_index=False).mean().sort_values(by=['Sex','Pclass'], ascending=True)
AgeGroup


# There seems to be some pattern in the table above. Passenger in higher class are generally older than those in the lower classes. Also avg age of men in each class is higher than in women.
# Also there are age values in decimals, which means there are infants on board, we should be able to identify them, but not with class or sex. 
# I would use the name feature here, since it has a title againt each passenger which can tell what type of person he/she is. Ex: Mr, Mrs, Miss, Master etc. Master is used to identify infant boys. 
# So i think its worth extracting the title feature.

# In[ ]:


# extracts the word before the string char "." this should give the salutation
allset['Salutation']=allset.Name.str.extract('([A-Za-z]+)\.', expand=False)
#grouping similar titles in one category
allset['Salutation'] = allset['Salutation'].replace(['Capt', 'Col','Major', 'Sir'], 'Officer')
#cleaning some titles which look like errors
allset['Salutation'] = allset['Salutation'].replace('Mlle', 'Miss')
allset['Salutation'] = allset['Salutation'].replace('Mme', 'Mrs')
allset['Salutation'] = allset['Salutation'].replace('Ms', 'Miss')
#grouping remaining titles into a last category called Rate
allset['Salutation'] = allset['Salutation'].replace(['Lady', 'Countess','Don', 'Dr','Rev','Jonkheer', 'Dona'], 'Rare')
print(allset['Salutation'].unique())


# Now i am going to use Sex, Pclass and Salutation together to create a table of avg ages for each group. I will then use this to fill up the missing values of age based on which category that passenger belongs to

# In[ ]:


#get all not null age data set
YesAge=allset[allset['Age']>0]
#create a table which stores the average age for each combination of Pclass, Sex and Salutation
AgeGroup=YesAge[['Age', 'Pclass','Sex','Salutation']].groupby(['Sex','Pclass','Salutation'], as_index=False).mean().sort_values(by=['Sex','Pclass','Salutation'], ascending=True)

#Iterate through the data set and fill up the empty age values using the table create above as reference
for Sex in allset.Sex.unique():
        for Class in allset.Pclass.unique():
            for Salutation in allset.Salutation.unique():
                if AgeGroup.loc[(AgeGroup.Sex==Sex) & (AgeGroup.Pclass==Class) & (AgeGroup.Salutation==Salutation),'Age'].count()>0:
                    allset.loc[ (allset.Age.isnull()) & (allset.Sex == Sex) & (allset.Pclass == Class) & (allset.Salutation == Salutation),'Age'] = AgeGroup.loc[(AgeGroup.Sex==Sex) & (AgeGroup.Pclass==Class) & (AgeGroup.Salutation==Salutation),'Age'].values[0]


# Lets bucket the age feature. Pandas library offers two functions cut and qcut to buckets age group. Lets see how they both look. We will bucket the age into 4 sets

# In[ ]:


allset['AgeBand']=pd.cut(allset['Age'],4)
Chart, items =plt.subplots(1,2,figsize=(25,5))
CAgeCut = sns.barplot(x="AgeBand",y="Survived",data=allset,ax=items[0])
CAgeCut = CAgeCut.set_ylabel("Survival Probability")
allset['AgeBand']=pd.qcut(allset['Age'],4)
CAgeQCut = sns.barplot(x="AgeBand",y="Survived",data=allset,ax=items[1])
CAgeQCut = CAgeQCut.set_ylabel("Survival Probability")


# Grouping the age into different buckets does seem to result in a more useful feature which can be used in our model, but i am not quiet convinced with the grouing. This is where apart from knowledge on Data Science concepts, one should also have the domain knowledge. From looking at the dataset, it looks like infant are likely to survive, then followed by young child. People older than 60 have the worst survival chance.
# so instead of using the pandas function, i would customize my age group like below and lets see how the chart looks after that

# In[ ]:


#Group age into 4 buckets
allset['AgeBand']=''
allset.loc[allset['Age']<=1,'AgeBand']='Infant'
allset.loc[(allset['Age']>1) & (allset['Age']<=10),'AgeBand']='YoungChild'
allset.loc[(allset['Age']>10) & (allset['Age']<=60),'AgeBand']='Adults'
allset.loc[allset['Age']>60,'AgeBand']='Seniors'

#plot bar chart to show the results of average survival
Chart, items =plt.subplots(1,1,figsize=(15,5))
CAgeCut = sns.barplot(x="AgeBand",y="Survived",data=allset,order=["Infant", "YoungChild", "Adults", "Seniors"])
CAgeCut = CAgeCut.set_ylabel("Survival Probability")


# Now this makes more sense. We can use this feature in our model. 
# I did notice the feature embarked also had 2 empty values. I wouldnt bother doing a detailed analysis to arrive at the missing value since its just two values. So lets just fill this empty cells with the highest occuring value in the whole dataset
# 

# In[ ]:


print("Null Embarked values: " , allset.Embarked.isnull().sum())
HighestBoarding=allset.Embarked.mode()[0]
print("Highest Boarding Point", HighestBoarding)
allset['Embarked']=allset['Embarked'].fillna(HighestBoarding)


# Alright, lets follow the same approach for Ticket, Sibsp and Parch. I will combine Sibsp and Parch to form Total Family feature. And for ticket i will use the first char to categorize tickets and check their survival probability on a chart.
# 

# In[ ]:


#Create a new feature to identify First Char of Ticket
allset['TicketFirst']=allset['Ticket'].str.slice(0,1)
Chart, items =plt.subplots(1,1,figsize=(15,5))
#Plot chart
CTicket = sns.barplot(x="TicketFirst",y="Survived",data=allset)
CTicket = CTicket.set_ylabel("Survival Probability")


# Like for Age, i will group Tickets into three buckets based on the survival probability

# In[ ]:


#Group Tickets into 3 buckets: High , Medium and Low
allset.loc[(allset['TicketFirst']=='5') | (allset['TicketFirst']=='8') | (allset['TicketFirst']=='A')| (allset['TicketFirst']=='7')| (allset['TicketFirst']=='W')| (allset['TicketFirst']=='6')| (allset['TicketFirst']=='4'),'TicketBucket']='Low'
allset.loc[(allset['TicketFirst']=='3') | (allset['TicketFirst']=='L') | (allset['TicketFirst']=='S')| (allset['TicketFirst']=='C'),'TicketBucket']='Medium'
allset.loc[(allset['TicketFirst']=='2') | (allset['TicketFirst']=='F') | (allset['TicketFirst']=='1')| (allset['TicketFirst']=='P')| (allset['TicketFirst']=='9'),'TicketBucket']='High'
Chart, items =plt.subplots(1,1,figsize=(15,5))
#Plot chart
CTicketBucket = sns.barplot(x="TicketBucket",y="Survived",data=allset,order=["High", "Medium", "Low"])
CTicketBucket = CTicketBucket.set_ylabel("Survival Probability")


# Do the same for Family. I will categorize family into Single, SmallFamily and LargeFamily.

# In[ ]:



allset['TotalFamily']=allset['SibSp'] + allset['Parch']
allset['FamilyBucket']='Single'
allset.loc[allset['TotalFamily']==0,'FamilyBucket']='Single'
allset.loc[(allset['TotalFamily']>0) & (allset['TotalFamily']<=3),'FamilyBucket']='SmallFamily'
allset.loc[allset['TotalFamily']>3,'FamilyBucket']='LargeFamily'
Chart, items =plt.subplots(1,1,figsize=(15,5))
#Plot chart
CFamilyBucket = sns.barplot(x="FamilyBucket",y="Survived",data=allset,order=["SmallFamily", "Single", "LargeFamily"])
CFamilyBucket = CFamilyBucket.set_ylabel("Survival Probability")


# As you can see, Small Family has higher chance of survival, followed by Singles and last Large Family.
# One another cool pattern i found. I must give other kernels to give this idea, but i slightly changed it. If you notice, if one family member survives, the chances of whole family surviving is high.
# So we will create a new feature called Family Survival. From the age bucket, it is clear that Infant and YoungChild have high survival. So if a infant or young child survives, i am assuming that the whoel family survives.
# I will use Family name to identify the whole family. Lets see how that looks

# In[ ]:


allset['LastName']=allset.Name.str.extract('([A-Za-z]+)\,', expand=False)

#WithFamily=allset.loc[(allset['Age']<=10) & (allset['Survived']==1),'LastName']
WithFamily=allset.loc[(allset['Age']<=10),'LastName']
allset['FamilySurvived']=0
for lastname in WithFamily:
    allset.loc[(allset['LastName']==lastname),'FamilySurvived']=1
print(allset.FamilySurvived.sum())


# Alright!!! with that we have identified all the features we need. I have ignored Fare, Cabin, PassengerID as from the analysis there isnt a good correlation and lot of noise(lot of empty values for cabin).
# Before i use the feature, i performed one improtant operation on the features above. **One Hot Encoding** . It can be well explained with an example. Take Pclass with possible values as 1, 2 and 3. Using the feature as in put to the model will result in class 3 getting higher weightage than task 1, where in reality class 1 as higher survival chance. so the value of the feature affects. To remove this, we would convert each feature into a one hot encoding where in for Pclass as an example, we would create 3 new feature with binary values(which identifies the presence or absence of this feature).
# Lets do that for all the feature below

# In[ ]:



# use pandas get_dummies function to generate one hot encoding for each feature
DClass=pd.get_dummies(allset['Pclass'], prefix='Pclass')
# append the new feature to the existing set
allset = pd.concat([allset, DClass], axis=1)
DEmbarked=pd.get_dummies(allset['Embarked'], prefix='Embarked')
allset = pd.concat([allset, DEmbarked], axis=1)
DTicketFirst=pd.get_dummies(allset['TicketBucket'], prefix='TicketBucket')
allset = pd.concat([allset, DTicketFirst], axis=1)    
DGender=pd.get_dummies(allset['Sex'], prefix='Sex')
allset = pd.concat([allset, DGender], axis=1)
DFamilyBucket=pd.get_dummies(allset['FamilyBucket'], prefix='FamilyBucket')
allset = pd.concat([allset, DFamilyBucket], axis=1)    
DAgeBand=pd.get_dummies(allset['AgeBand'], prefix='AgeBand')
allset = pd.concat([allset, DAgeBand], axis=1)    


# So that is done. Finally lets drop all the features we don need anymore and see how the final data set looks...

# In[ ]:


#Drop features
allset=allset.drop(['Name'],axis=1)
allset=allset.drop(['Age'],axis=1)
allset=allset.drop(['AgeBand'],axis=1)
allset=allset.drop(['Ticket'],axis=1)
allset=allset.drop(['TicketFirst'],axis=1)
allset=allset.drop(['Fare'],axis=1)
allset=allset.drop(['Pclass'],axis=1)
allset=allset.drop(['Salutation'],axis=1)
allset=allset.drop(['Cabin'],axis=1)
allset=allset.drop(['Embarked'],axis=1)
allset=allset.drop(['Sex'],axis=1)
allset=allset.drop(['SibSp'],axis=1)
allset=allset.drop(['Parch'],axis=1)
allset=allset.drop(['TotalFamily'],axis=1)
allset=allset.drop(['LastName'],axis=1)
allset=allset.drop(['FamilyBucket'],axis=1)
allset=allset.drop(['TicketBucket'],axis=1)

train_df=allset[:891]
test_df=allset[891:]
train_df.head(20)


# Lets create the train, output and test data set.

# In[ ]:


#Remove Survived and PassengerID for XTrain
X_Train=train_df.drop(['Survived','PassengerId'],axis=1)
#YTrain will contain the output feature Survived
Y_Train=train_df.Survived
#Create test set which is same as X_Train but without Survived
X_Test=test_df.drop(['PassengerId','Survived'],axis=1)
X_Train.shape,Y_Train.shape,X_Test.shape


# Lets do the modelling. I initiall selected LogisticRegression and it gave me an accuracy of 83.04 on the train set, but on the test set, it gave 79.4. I was more keen on hitting the 80% mark. I hence used RandomForestClassifier. Its an ensemble model using Decision Trees where in it create multiple Decision Trees and uses voting to determine the best prediction.

# In[ ]:


logreg = RandomForestClassifier()
logreg.fit(X_Train, Y_Train)
Y_Pred = logreg.predict(X_Test)
acc_log = round(logreg.score(X_Train, Y_Train) * 100, 2)
acc_log


# The last step is to create the output file and submit it to Kaggle. I got a accuracy of 81.8% on the test data set and put me on the rank 427!!!(Top 5%)
# 

# In[ ]:


FinalResult=pd.DataFrame({'PassengerId':test_df["PassengerId"],'Survived':Y_Pred.astype(int)})
FinalResult.to_csv('gender_submission.csv', index=False)
TestResult=pd.DataFrame(test_df)


# In[ ]:





# In[ ]:





# In[ ]:




