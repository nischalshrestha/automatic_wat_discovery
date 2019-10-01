#!/usr/bin/env python
# coding: utf-8

# *UPDATE 11-21-2018*
# 
# WIP : I added a new feature based on Ticket and I measure the mean survival rate of people having close ticket numbers. 
# I will sum up and fully automate the classification task using xgboost and these new features in the coming days. 

# 1. [Introduction](#Intro)
# 2. [References](#Ref)
# 3. [Model](#Model)
# <br/>
#  3.1[Dummy Sex Model](#Dummy)
# <br/>
#  3.2[Feature Engineering](#FEng)
# <br/>
#  3.3[Mothers & Babies](#MB)
# <br/>
#  3.4[Other Groups](#G)
# <br/>
#  3.5[Single & Isolated Women](#SW)
# 4. [Further Improvements](#FI)
# 5. [Checks](#C)
# 6. [CSV Output](#CSV)
# 
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

# machine learning
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
#from sklearn.compose import ColumnTransformer not present in version 19.1
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import mean_squared_error, precision_score,precision_recall_curve,recall_score,f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder, LabelBinarizer
from sklearn.impute import SimpleImputer


# In[ ]:


dftrain=pd.read_csv('../input/train.csv')
dftest=pd.read_csv('../input/test.csv')
dftrain.head(3)


# <h1><a id=Intro>Introduction</a>

#  ### 1. Common mistake
# 
# I had a lot of fun exploring the Titanic dataset and felt like an investigator in charge of discovering the reasons why some people survived and other died. I think that this competition is a great way to start learning but teaches you the harsh way !  Indeed, as most of us, I first blindly applied my favorite classification algorithm, that is Random Forest, barely looking at the features. I was mainly focused on tweaking hyperparameters and designing very complex strategies to fill in missing values for Age, Fare, Embarked... I could not overtop the 79-80% of the top 20% of participants. 
#  
#  ### 2. Simple is beautiful
#  
# Until I accidentally discovered that some people in the test set shared family bounds with people in the train set. Everything became clear at that moment. I finally understood that it did not matter if you were 35 or 40 or if you paid 12£ instead of 20£. What really matters here is the fate of people whom you were travelling with. Indeed, in all this chaos, what prevailed was bounds between passengers.
# 
# ### 3. Build a transparent model
# 
# So the challenge is to properly identify people who were travelling together and decide on the fate of people in the test set given what happened to the people in the train set that belonged to the same group. **We then build here our own decision tree that is a transparent model that we fully understand. **
#  
# Thus I want to insist on a particular point of interest that contradicts what we usually learn doing Machine Learning that we should not use any information on the test set while building our model. If we follow this principle here and run classical algorithms on the training set, they won't be able to see the connections that could exist between the training and test sets. As you could have noticed, by naively training classification algorithms, they often output that all men died. Slightly better, they output that all youg Masters survived even if they have mothers who died in the training set.  That is why we should manually correct them or simply don't use them here as the most important features are neither quantitative nor categorical as Name or Ticket. 

# <h1><a id=Ref>References</a>

# An excellent notebook that hugely influenced mine : 
# [Chris Deotte - Titanic WCG + XGBoost](https://www.kaggle.com/cdeotte/titanic-wcg-xgboost-0-84688).
# It is the best scoring notebook with detailed explanations. Overperforming notebooks are only copy-paste of the black-box genetic programming original code. 
# 
# [Aurélien Géron - Hands-On Machine Learning with Scikit-Learn and TensorFlow](https://www.amazon.fr/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1491962291).
# A perfect place to start learning Machine Learning with neat python code relying on two of the most used libraries Sklearn and TF. 
# In particular, it has a dedicated section to classification and clearly explains why it is more difficult to design adequate metrics than for regression. You could find here definitions for the different metrics I use to evalute the Dummy Sex Model. 
# 
# 
# 

# <h1><a id=Model>Model</a>

# ## Creating a distance metric splitting Ticket numbers

# The idea here is again to regroup passengers with the people they were travelling with and identify groups of people who all died. This may be because they were poor, located in a bad area of the boat... To do so, I convert Ticket feature to a ticket number and I define the group surrounding a person with the passengers having the same ticket +/- k=10.
# 
# TODO : I should cross-validate k on the training set to fine tune the value of this hyperparameter

# In[ ]:


def ticket2(value):
    split=value.split(" ")
    if(len(split)>1):
        return int(split[-1].strip())
    else :
        return 0


# In[ ]:


dftotal=pd.concat([dftrain,dftest],axis=0,ignore_index=True)
dftotal["Ticket1"]=dftotal.Ticket.map(lambda x: x.split(" ")[0])
dftotal["Ticket2"]=dftotal.Ticket.map(ticket2)
#dftotal["Ticket2"]=dftotal["Ticket2"].astype(int)


# -KNNs contains a list of all the passengers index surrounding the individual. Very useful to view and debug the function below. 
# -KNNsNbr just a count of KNNs
# -KSurvivalRate is the survival rate of the people surrounding the person. 

# In[ ]:


dftotal["KNNs"]=""
dftotal["KNNsNbr"]=0
dftotal["KSurvivalRate"]=0


# In[ ]:


k=10
for index in dftotal.index:
    ticket1=dftotal.loc[index,"Ticket1"]
    ticket2=dftotal.loc[index,"Ticket2"]
    embarked = dftotal.loc[index,"Embarked"]
    if(ticket1=="LINE"):
        nbours_index=dftotal[dftotal.Ticket=="LINE"].index
        dftotal.at[index,"KNNs"]=nbours_index.values
        dftotal.loc[index,"KNNsNbr"]=len(nbours_index)
        dftotal.loc[index,"KSurvivalRate"]=dftotal.loc[nbours_index,"Survived"].mean()
    elif((ticket1!="LINE")&(ticket2==0)):
        dftotalindex=dftotal[(dftotal.Embarked==embarked)&(dftotal.Ticket2==0)&(dftotal.Ticket!="LINE")].copy()
        nbours_index=dftotalindex[np.abs(dftotalindex["Ticket1"].astype(int)-int(ticket1))<k].index
        dftotal.at[index,"KNNs"]=nbours_index.values
        dftotal.loc[index,"KNNsNbr"]=len(nbours_index)
        dftotal.loc[index,"KSurvivalRate"]=dftotalindex.loc[nbours_index,"Survived"].mean()
    else:
        dftotalindex=dftotal[(dftotal.Embarked==embarked)&(dftotal.Ticket1==ticket1)].copy()
        nbours_index=dftotalindex[np.abs(dftotalindex["Ticket2"].astype(int)-int(ticket2))<k].index
        dftotal.at[index,"KNNs"]=nbours_index.values
        dftotal.loc[index,"KNNsNbr"]=len(nbours_index)
        dftotal.loc[index,"KSurvivalRate"]=dftotalindex.loc[nbours_index,"Survived"].mean()


# ### Women of third class surrounded by people who all died

# In[ ]:


(dftotal[(dftotal.KNNsNbr>5)&(dftotal.KSurvivalRate==0.0)
        &(dftotal.PassengerId>891)&(dftotal.Sex=="female")&(dftotal.Pclass==3)])


# ### Storing their PassengerIds

# In[ ]:


women_group_dead=dftotal[(dftotal.KNNsNbr>0)&(dftotal.KSurvivalRate==0.0)
        &(dftotal.PassengerId>891)&(dftotal.Sex=="female")&(dftotal.Pclass==3)]["PassengerId"].values
women_group_dead


# <h2><a id=Dummy>Dummy Sex Model</a>

# The first obvious observation is that **Women survived and Men died** with a slight distinction between Pclasses. That will be our default model that we build here using Sklearn framework in order to evaluate it on different training samples with various metrics. 

# In[ ]:


group=dftrain.groupby(["Sex","Pclass"])
group["Survived","Age","Fare"].agg(["mean","std","size"])


# We can see on the table above that we are dealing with and **unbalanced classification problem**. That is a large majority of women survived and men died. That is one of the reason why applying blindly classification algos won't work. Still they will have a good accuracy as the naive strategy of saying that everyone died gets already a good score. 

# In[ ]:


from sklearn.base import BaseEstimator

class DummyPrediction(BaseEstimator):
    
    def fit(self, X, Y=None):
        pass
    
    def predict(self, X):
        Y=X["Sex"]=="female"
        return Y.astype(int)
            


# In[ ]:


def print_scores(scores):
    print(scores)
    print(scores.mean())
    print(scores.std())


# In[ ]:


clf = DummyPrediction()
dftrain["Survived_Model"]=clf.predict(dftrain)


# In[ ]:


Y=dftrain["Survived"]
Y_model=dftrain["Survived_Model"]
print("recall score on training set",recall_score(Y,Y_model))


# In[ ]:


print("precision score on training set",precision_score(Y,Y_model))


# In[ ]:


scores=cross_validate(clf,dftrain,dftrain["Survived"],scoring=["f1","accuracy"],cv=10,return_train_score=False)


# In[ ]:


def display_cross_validate(scores):
    print("cross val scores")
    print("f1 scores", scores["test_f1"])
    print("f1 mean", scores["test_f1"].mean())
    print("f1 std", scores["test_f1"].std())
    print("accuracy scores", scores["test_accuracy"])
    print("accuracy mean", scores["test_accuracy"].mean())
    print("accuracy std", scores["test_accuracy"].std())


# In[ ]:


display_cross_validate(scores)


# In[ ]:


print("roc_auc_score on training set",roc_auc_score(Y,Y_model))


# ## Default Prediction
# 
# Except for "outliers" that we will try to identify below, our default prediction will be that women survived and men died. We get a reasonable score on the public test set of 76.55%. Then to refine our model we  will use different features and feature engineering to identify rare women that should have died and rare men that survived.

# In[ ]:


dftest["Survived"]=clf.predict(dftest)


# <h2><a id=FEng>Feature Engineering </a>

# ## Processing Name

# The goal here is to extract from Name, the family name, title and last two letters of family name. Here the idea is to classify people according to their nationality. We will see below that this last feature can identify people from Yougoslavia, Russia for instance. 

# In[ ]:


def process_names(df):
    df["FamilyName"]=df["Name"].map(lambda x: x.split(",")[0].strip())
    df["FullName"]=df["Name"].map(lambda x: x.split(",")[1].strip())
    df["Title"]=df["FullName"].map(lambda x: x.split(".")[0].strip())
    df["TwoLetters"]=df["FamilyName"].map(lambda x: x[-2:])
    
process_names(dftrain)
process_names(dftest)


# ## Processing Ticket

# **Ticket entries are not unique**. We want to identify duplicates. By doing so we will be able to spot large families, groups and be able to correct Fare to get the real Fare per passenger. We also create a new column with the Ticket number without the last figure so as again to spot groups. 

# In[ ]:


print("most occuring tickets in the training set:")
dftrain["Ticket"].value_counts()[:6]


# In[ ]:


dftrain["TicketButLast"]=dftrain.Ticket.map(lambda x: x[:-1])
dftest["TicketButLast"]=dftest.Ticket.map(lambda x: x[:-1])


# Then I create new columns to count the number of occurences for each ticket number as follows : 
# * for tickets in train :
# * * the number of duplicates in train, in train & test combined if intersection not empty. 
# * for tickets in test : 
# * * the number of duplicates in test, in train & test combined if intersection not empty. 
# 
# We do the same for the ticket number without the same figure. As groups travelling together but not sharing family bounds or buying their tickets individually could not be spotted otherwise. 
# People from Yougoslavia ex : 

# In[ ]:


dftrain[dftrain.TwoLetters=="ic"].sort_values("Ticket")[["PassengerId","Survived","Pclass","Name","Ticket"]]


# In[ ]:


ticket_count_train = dftrain["Ticket"].value_counts()
ticket_count_test = dftest["Ticket"].value_counts()
ticket_inter = np.intersect1d(dftrain["Ticket"].values,dftest["Ticket"].values)

ticketButLast_count_train = dftrain["TicketButLast"].value_counts()
ticketButLast_count_test = dftest["TicketButLast"].value_counts()
ticketButLast_inter = np.intersect1d(dftrain["TicketButLast"].values,dftest["TicketButLast"].values)


# In[ ]:


for idx in dftrain.index:
    ticket = dftrain.loc[idx,"Ticket"]
    dftrain.loc[idx,"CountTicket_InTrain"]=ticket_count_train[ticket]
    if(ticket in ticket_inter):
        dftrain.loc[idx,"CountTicket"]=(ticket_count_train[ticket]+ticket_count_test[ticket])
    else:
        dftrain.loc[idx,"CountTicket"]=ticket_count_train[ticket]


# In[ ]:


for idx in dftest.index:
    ticket = dftest.loc[idx,"Ticket"]
    dftest.loc[idx,"CountTicket_InTest"]=ticket_count_test[ticket]
    if(ticket in ticket_inter):
        dftest.loc[idx,"CountTicket"]=(ticket_count_train[ticket]+ticket_count_test[ticket])
    else:
        dftest.loc[idx,"CountTicket"]=ticket_count_test[ticket]


# In[ ]:


for idx in dftrain.index:
    ticketButLast = dftrain.loc[idx,"TicketButLast"]
    dftrain.loc[idx,"CountTicketButLast_InTrain"]=ticketButLast_count_train[ticketButLast]
    if(ticketButLast in ticketButLast_inter):
        dftrain.loc[idx,"CountTicketButLast"]=(ticketButLast_count_train[ticketButLast]
                                               +ticketButLast_count_test[ticketButLast])
    else:
        dftrain.loc[idx,"CountTicketButLast"]=ticketButLast_count_train[ticketButLast]


# In[ ]:


for idx in dftest.index:
    ticketButLast = dftest.loc[idx,"TicketButLast"]
    dftest.loc[idx,"CountTicketButLast_InTest"]=ticketButLast_count_test[ticketButLast]
    if(ticketButLast in ticketButLast_inter):
        dftest.loc[idx,"CountTicketButLast"]=(ticketButLast_count_train[ticketButLast]
                                               +ticketButLast_count_test[ticketButLast])
        dftest.loc[idx,"CountTicketButLast_InTrain"]=(ticketButLast_count_train[ticketButLast])
    else:
        dftest.loc[idx,"CountTicketButLast"]=ticketButLast_count_test[ticketButLast]


# ## Obtaining correct Fare per passenger

# In[ ]:


for idx in dftrain.index:
    ticket = dftrain.loc[idx,"Ticket"]
    if(ticket in ticket_inter):
        dftrain.loc[idx,"FareCorrect"]=dftrain.loc[idx,"Fare"]/(ticket_count_train[ticket]
                                                                +ticket_count_test[ticket])
    else:
        dftrain.loc[idx,"FareCorrect"]=dftrain.loc[idx,"Fare"]/(ticket_count_train[ticket])


# In[ ]:


print("Example of correction for large Pclass 3 Family")
dftrain[dftrain.FamilyName=="Panula"][["Survived","Pclass","Name","Ticket","Fare","FareCorrect"]]


# <h2><a id=MB>Mothers & Babies </a>

# ## Mothers won't abandon their children

# First find mothers in the test set that have babies or family members who died in the training set. We use FamilyName and Age to identify those women.  

# In[ ]:


dfFamilyTrain=dftrain[(dftrain["Parch"]>0)&(dftrain["Survived"]==0)]
dfFamily=dfFamilyTrain[(dfFamilyTrain["Sex"]=="female")|(dfFamilyTrain["Age"]<10)]
familiestrain=dfFamily["FamilyName"]


# In[ ]:


dfFamilytest=dftest[(dftest["Parch"]>0)&(dftest["Sex"]=="female")]
familiestest=dfFamilytest["FamilyName"]


# In[ ]:


print("Women who have family members who died in the training set.")
intersection=np.intersect1d(familiestest,familiestrain)
intersection


# In[ ]:


women_died_ids=dfFamilytest[dfFamilytest["FamilyName"].isin(intersection)]["PassengerId"].index


# In[ ]:


col_index=dftest.columns.get_loc("Survived")
dftest.iloc[women_died_ids,col_index]=0


# In[ ]:


print("Mothers and Sisters who died with their families")
dftest.iloc[women_died_ids,:]


# With this correction my submission score went from 76.55% to 78.46% which is equal to an improvement of 1.91%=4/(417/2). So I think that 4 out of the 7 cases we found above are contained in the test set used to compute public initial scores. 

# ## Boys who survived with their mothers

# To identify them we select males with Title=="Master" and look for their mums in the training set based on FamilyName.

# In[ ]:


dfFamilyTrain=dftrain[(dftrain["Parch"]>0)&(dftrain["Survived"]==1)]
dfFamily=dfFamilyTrain[(dftrain["Sex"]=="female")]
familiestrain=dfFamily["FamilyName"]


# In[ ]:


dfFamilytest=dftest[(dftest["Parch"]>0)&(dftest["Sex"]=="male")&((dftest["Title"]=="Master"))]
familiestest=dfFamilytest["FamilyName"]


# In[ ]:


print("Family names of Boys with mothers who survived ")
intersection=np.intersect1d(familiestest,familiestrain)
intersection


# In[ ]:


boys_survived_ids=dfFamilytest[dfFamilytest["FamilyName"].isin(intersection)]["PassengerId"].index


# In[ ]:


col_index=dftest.columns.get_loc("Survived")
dftest.iloc[boys_survived_ids,col_index]=1


# In[ ]:


print("Boys who survived with their mothers")
dftest.iloc[boys_survived_ids,:]


# ## Boy with non family members and mother alone

# Here I just check for "Masters" for which I did not find mothers in the training set. Some could be in the test set or they could travel with adults. 

# In[ ]:


dftest[(dftest.Title=="Master")&(dftest.Survived==0)]


# I checked manually for all of them using FamilyName, Ticket and TicketButLast. For Rice, Boulos, van Billiard, Dangbom, Johnston, Betros, Sage, Palsson family died. For Olsen his father died but I found that another Norwegian travelling with them survived so I made the hypotheses that he saved the kid. 

# In[ ]:


print("Male passenger travelling with Olsen, Master. Artur Karl who survived:")
dftrain[dftrain.Ticket.map(lambda x: x[:-1])=="C 1736"]


# For Peacock, Master. Alfred Edward I saw that his mother and sister were also in the test set with no other family member. So I presumed that a poor mother with 2 young kids died : 

# In[ ]:


dftest[dftest.FamilyName=="Peacock"]


# ## Olsen survived and Peacock family died

# In[ ]:


olsen_idx=dftest[(dftest.FamilyName=="Olsen")].index
dftest.loc[olsen_idx,"Survived"]=1
peacock_idx=dftest[(dftest.FamilyName=="Peacock")].index
dftest.loc[peacock_idx,"Survived"]=0


# <h2><a id=G>Other Groups </a>

# ## Same Ticket

# By looking at most occuring tickets I could see that some people were travelling with families as "au paire" or groups of friends from the same regrion. 

# In[ ]:


print("Panula family Ticket number 3101295")
dftrain[dftrain.Ticket=="3101295"]


# In[ ]:


print("Travelling with them in test set")
dftest[dftest.Ticket=="3101295"]


# In[ ]:


print("Most occuring ticket in train set, a group of Chinese. Most of them survived.")
dftrain[dftrain.Ticket=="1601"]


# In[ ]:


print("accompanying them in test set : ")
dftest[dftest.Ticket=="1601"]


# In[ ]:


most_occ_ticks=dftrain["Ticket"].value_counts()[:19]


# In[ ]:


df=dftest[dftest["Ticket"].isin(most_occ_ticks.index)].sort_values("Ticket")
df_women=df[df["Sex"]=="female"]


# In[ ]:


print("Women who died with their groups. We already spotted some mothers above.")
for idx in df_women.index:
    ticket = df_women.loc[idx,"Ticket"]
    nbr_survivors = dftrain[dftrain["Ticket"]==ticket]["Survived"].sum()
    if(nbr_survivors==0):
        print(df_women.loc[idx,"Name"])
        dftest.loc[idx,"Survived"]=0


# In[ ]:


df_men=df[(df["Sex"]=="male")&(df["SibSp"]==0)&(df["Parch"]==0)]
df_men


# In[ ]:


print("Man who survived with the majority of his group:")
for idx in df_men.index:
    ticket = df_men.loc[idx,"Ticket"]
    mean_survivors = dftrain[dftrain["Ticket"]==ticket]["Survived"].mean()
    if(mean_survivors>0.5):
        print(df_men.loc[idx,"Name"])
        dftest.loc[idx,"Survived"]=1


# ## Same TicketButLast

# In the same idea as above, we look for young women from Pclass3 who could have accompanied large families who died and took care of their children. 

# In[ ]:


dftest[(dftest.CountTicketButLast>10)&(dftest.Pclass==3)&(dftest.SibSp==0)&(dftest.Parch==0)&(dftest.Sex=="female")&
      (dftest.Title=="Miss")]


# We observe that Nieminen, Miss. Manta Josefina should have travelled with the Panula family and their 6 children. They all died. On the table below only Hirvonen, Miss. Hildur E, 2 years old survived. Her mother is in the test set. 

# In[ ]:


print("Passengers of train set who travelled with Nieminen, Miss. Manta Josefina")
dftrain[dftrain.TicketButLast=="310129"]


# In[ ]:


print("Passengers of train set who travelled with Nieminen, Miss. Jenny Lovisa")
dftrain[dftrain.TicketButLast=="34708"]


# In[ ]:


idx=dftest[(dftest.CountTicketButLast>10)&(dftest.Pclass==3)&(dftest.SibSp==0)&(dftest.Parch==0)&(dftest.Sex=="female")&
      (dftest.Title=="Miss")].index
dftest.loc[idx,"Survived"]=0


# ## People from Yougoslavia and Russia died

# Looking at the last two letters of FamilyName is a proxy to spot people from the same regions. It is a way to clusterize people. 

# In[ ]:


print("People called ---ic died :")
dftrain[dftrain.TwoLetters=="ic"]


# In[ ]:


print("People called ---ff died :")
dftrain[dftrain.TwoLetters=="ff"]


# Looking for similar people in the test set we found two women. Another way we could have spotted was by looking at their ticket number since they have consecutive ticket numbers. 

# In[ ]:


dftest[(dftest.TwoLetters.isin(["ic","ff"]))&(dftest.Sex=="female")]


# In[ ]:


idx=dftest[(dftest.TwoLetters.isin(["ic","ff"]))&(dftest.Sex=="female")].index
dftest.loc[idx,"Survived"]=0


# ## Baron von Drachstedt

# Here looking at the names of men in the test set and using the split method, we luckily found that one had a noble title. We decided to assume that he survived. 

# In[ ]:


dftest[dftest["PassengerId"]==1297]


# In[ ]:


dftest.loc[dftest[dftest["PassengerId"]==1297].index,"Survived"]=1


# <h2><a id=SW>Single & Isolated Women </a>

# In[ ]:


print("single women statistics")
dftrain[(dftrain["Sex"]=="female")&(dftrain["SibSp"]==0)&(dftrain["Parch"]==0)].describe()


# Here I use very cautiously the xgboost classifier that I train only on single women. By doing so I try to eliminate all the biases that come from the human bounds we have studied above. I then only select the 5 worst scores that are very close to 0 and assume that those women die. Moreover as for the single women training set, 40% of women survived. So we are training the algo on a **balanced data set**. 

# # ALGORITHMIC CLASSIFICATION for SINGLE WOMEN

# In[ ]:


single_women = dftrain[(dftrain["Sex"]=="female")&(dftrain["SibSp"]==0)&(dftrain["Parch"]==0)]
single_women["Age"]=single_women["Age"].fillna(single_women["Age"].median())
features=["Age","Pclass","Fare"]
x_single_women = single_women[features]
y_single_women = single_women["Survived"]


# In[ ]:


xg = GradientBoostingClassifier()
kfold = StratifiedKFold(10)


# In[ ]:


param_grid={
    'n_estimators':[100,200,300],
    'min_samples_split':[2,3,4],
    'max_depth' : np.arange(3,7,1)
}


# In[ ]:


grid = GridSearchCV(xg,param_grid=param_grid,cv=kfold,scoring="accuracy")
grid.fit(x_single_women,y_single_women)


# In[ ]:


grid.best_params_


# In[ ]:


test_df=dftest[(dftest["Sex"]=="female")&(dftest["SibSp"]==0)&(dftest["Parch"]==0)]
test_df["Age"]=test_df["Age"].fillna(test_df["Age"].median())
test_df["Fare"]=test_df["Fare"].fillna(test_df["Fare"].median())


# In[ ]:


Y_scores_pclass3 = grid.best_estimator_.predict_proba(test_df[features])[:,1]
test_df["Y_scores"]=Y_scores_pclass3


# In[ ]:


print("Five worst scores of xgboost :")
test_df.sort_values("Y_scores",ascending=True)[:5]


# We had already found that Jelka Oreskovic should have died. We found 4 other women very likely to die. 

# In[ ]:


idxs = test_df.sort_values("Y_scores",ascending=True).index[:5]
dftest.loc[idxs,"Survived"]=0


# ## Isolated women

# In[ ]:


single_poor_wmn=dftest[(dftest.CountTicketButLast<=2)&(dftest.Sex=="female")
                       &(dftest.Pclass==3)&(dftest.SibSp==0)&(dftest.Parch==0)].index
dftest.loc[single_poor_wmn,"Survived"]=0


# In[ ]:


print("Single poor isolated women with no family")
dftest.loc[single_poor_wmn,:]


# In[ ]:


idx=dftest[dftest.PassengerId.isin(women_group_dead)].index
dftest.loc[idx,"Survived"]=0


# <h2><a id=FI>Further Improvements </a>

# Manifestly, I did not find enough information regarding men who survived. I did not manage yet to do the same as I did for single women. That is finding a good, balanced training set to train a classification algo that would output very high survival probability for men in the test set that are not young Masters. 
# 
# Moreover, many operations in this notebook are done manually. I could improve it by creating real column features as for Mother, Father, Mother who died, Mother who Survived, Mean of Survival for People in the same group... Then feed those new features to a classification algorithm. 
# 
# Besided, more could be done also on the Name feature to extract the nationality of passengers. As we saw, some people travelled together, not sharing any family bounds based on family names, and shared the same fate. From the training data, we could aslo extrapolate the fact that non-english speaking passengers could be more likely to die. 
# 
# **I would be happy if some Kagglers forked this notebook and improve from it. I Welcome also all your comments to improve this code and I will add some comments if some operations were not clear enough.**

# <h2><a id=C>Checks </a>

# In[ ]:


print("Men who are predicted to survive")
dftest[(dftest["Survived"]==1)& (dftest["Sex"]=="male")]


# In[ ]:


print("Women who are predicted to survive")
dftest[(dftest["Survived"]==0)& (dftest["Sex"]=="female")]


# <h2><a id=CSV>CSV OUTPUT </a>

# In[ ]:


dftest_sorted=dftest.sort_values("PassengerId")
dftest_sorted=dftest_sorted[["PassengerId","Survived"]]


# In[ ]:


dftest_sorted.to_csv("Titanic.csv",index=False)
print('print csv')

