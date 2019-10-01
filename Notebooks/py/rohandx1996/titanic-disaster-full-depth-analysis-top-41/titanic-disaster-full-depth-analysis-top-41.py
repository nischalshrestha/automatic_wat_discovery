#!/usr/bin/env python
# coding: utf-8

# # Titanic In depth Analysis , my aims will be like getting insigts from stats test , some useful knowledge to show and atlast which machine learning algorithm is best and why graphs , learning curves visualization , and most common facing Data Leaking will not be happen , which 

# # how story TIMELINE WILL WORK 

# # OSEMN approach known as awesome , readers when read this notebook will get to know about this approach 

# # hence it was said that it will never sink but after more than 100 years data sciencetist from around the gathering to predict this passenger will survive or not ? on basis various features that passenger posses 

# In[ ]:


# importing  necesary files & libraries 


# In[ ]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.stats as ss
from statsmodels.formula.api import ols
from scipy.stats import zscore
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns
get_ipython().magic(u'timeit')
get_ipython().magic(u'matplotlib inline')


# In[ ]:


dftrain=pd.read_csv("../input/train.csv")
dftest=pd.read_csv("../input/test.csv")
test=dftest.copy()


# In[ ]:


dftrain.info()


# In[ ]:


dftrain.head()


# # Now go for analysis for features and there data types 

# In[ ]:


dftrain.info()


# # How to deal with null values as we can see cabin , AGE , EMBARKED are having most null values 

# # and conversion of object data type to categorical is necessary for reducing the space in memory and decrease  time in computation 

# In[ ]:


# analysis of dtypes 


# In[ ]:


plt.figure(figsize=(5,5))
sns.set(font_scale=2)
sns.countplot(y=dftrain.dtypes ,data=dftrain)
plt.xlabel("count of each data type")
plt.ylabel("data types")
plt.show()


# # as we can see that object data types are more and with equal to int , can increase computation time , need conversion in cat.dtype

# # nullity analysis 

# In[ ]:


import missingno as msno


# In[ ]:


msno.bar(dftrain.sample(890))


# # now nultity correlation wehave to see between age , cabin and embarked

# In[ ]:


msno.matrix(dftrain)


# In[ ]:


msno.heatmap(dftrain)


# In[ ]:


msno.dendrogram(dftrain)


# # our finding says that when cabin and age values will come and will be null together where as in case of emabarked it is reverse 

# # from this we are concluding a fact that only 38.8 % people survived , and even most young people died in this disaster about age of 30 

# # Now a pie chart percentage of Categories of people travelling survived 

# In[ ]:


df=dftrain.copy()
df.head()


# In[ ]:



male1=df.loc[(df.Survived==1) &(df.Sex=='male'),:].count()
female1=df.loc[(df.Survived==1) & (df.Sex=='female'),:].count()


# In[ ]:


print(male1)


# In[ ]:


print(female1)


# # so female survived more in than male since , during disaster , females are send from the ship first and siblings too. 

# In[ ]:


sns.factorplot(x="Sex",col="Survived", data=df , kind="count",size=6, aspect=.7,palette=['crimson','lightblue'])
malecount=pd.value_counts((df.Sex == 'male') & (df.Survived==1))
femalecount=pd.value_counts((df.Sex=='female') & (df.Survived==1))
totalmale,totalfemale=pd.value_counts(df.Sex)
print("male survived {} , female survived {}".format(malecount/totalmale,femalecount/totalfemale))


# # 18% of male survived and 74% percent female survived 

# In[ ]:


plt.figure(figsize=(12,12))
sns.swarmplot(x="Sex",y="Age",hue='Pclass',data=df,size=10 ,palette=['pink','lightgreen','purple'])


# # swarmplot inferences - 
# # 1> like there age group from 0 to 10 , 20 to 40 , 40 above till 52 then 55 to 65 then most likely grand parents
# 
# # 2> males are comparetively less in 1st class than female , females of higher class were most of them alone or not couple , travelling with friends , very few couples
# # 3> couples are very less since male red marks are more , so we can these were mostly labours , small kids, bachelors and they were in class 3 passengers
# # 4> most aged group were more in special 1st class than  2nd class

# # now inference with respect to survivebiltity and age factor

# In[ ]:


plt.figure(figsize=(12,12))
sns.swarmplot(x="Sex",y="Age",hue='Survived',data=df,size=10)


# # Most of the youth male died , siblings survived (both) , only i can say that high class passengers in male might got save 

# In[ ]:



sns.factorplot(x="Sex", hue = "Pclass" , col="Survived", data=df , kind="count",size=7, aspect=.7,palette=['crimson','orange','lightblue'])


# ### overall the males and females of Pclass 3 died more than others
# ### the males of Pclass 3 showed a remarkable increase in death and shoots the graph up , same goes to the females in
# ### same goes to the females in survived = 0
# ### in survived = 0 , showing increasing trend in death as class shifts down
# ### In survived = 1 females showed a near fall down trend as expected but pclass=2 females survived less than the Pclass=3 females
# ### But the males on contrary showed a dip in between i.e.
# ### in males who survived , Plass --> 3 > 1 > 2
# ### i.e Survived Pclass=3 males survived more than the survived Pclass=1 males and survived Pclass=2 males
# ### the above is evident from the following inspection

# In[ ]:


pd.crosstab([df.Sex,df.Survived],df.Pclass, margins=True).style.background_gradient(cmap='autumn_r')


# # so we can males which are less suvived were lower class people , labours , and bachelors but in first class as we can see out 77 male  45 sruvived nearly 60 percent  , money vs humanity , sad reality money wins always

# # now we focus on which group from each Passenger class survived more and less 
# 

# In[ ]:


pd.crosstab([df.Survived,df.Pclass],df.Age,margins=True).style.background_gradient(cmap='autumn_r')


# # kids on 1st class was one found and died , even 1st class children weren't there 
# # richer famlies were without chidren and even in that too many were bachelors.
# # 28 children were died in Plcass 3 and children 1st class 99% were saved , in second class 100% saved 
# # youth from 3 class died more than any class

# In[ ]:


sns.factorplot(x="Survived",col="Embarked",data=df ,hue="Pclass", kind="count",size=8, aspect=.7,palette=['crimson','darkblue','purple'])


# In[ ]:


pd.crosstab([df.Survived],[df.Sex,df.Pclass,df.Embarked],margins=True).style.background_gradient(cmap='autumn_r')


# # Embarked  S survived more in female in every passenger class, in male also same the same 
# # Least is from Q Embarked from both the case

# In[ ]:


sns.factorplot(x="Sex", y="Survived",col="Embarked",data=df ,hue="Pclass",kind="bar",size=7, aspect=.7)


# #  Embarked Analysis
# most male survived from emabarked C > S > Q
# 
# from  Q emabarked on 3 class passengers werae travelling and only 20 % were got saved and Q emabarked is the only were females from 1 class and 2 class not died 

# # Correlation Analysis 

# In[ ]:


context1 = {"female":0 , "male":1}
context2 = {"S":0 , "C":1 , "Q":2}
df['Sex_bool']=df.Sex.map(context1)
df["Embarked_bool"] = df.Embarked.map(context2)
plt.figure(figsize=(20,20))
correlation_map = df[['PassengerId', 'Survived', 'Pclass', 'Sex_bool', 'Age', 'SibSp',
       'Parch', 'Fare' , 'Embarked_bool']].corr()
sns.heatmap(correlation_map,vmax=.7, square=True,annot=True,fmt=".2f")


# # The above heatmap shows the overall picture very clearly
# ### PassengerId is a redundant column as its very much less related to all other attributes , we can remove it .
# ### Also , Survived is related indirectly with Pclass and also we earlier proved that as Pclass value increases Survival decreases
# ### Pclass and Age are also inversely related and can also be proven by the following cell that as Pclass decreases , the mean of the Age increases , means the much of the older travellers are travelling in high class .
# ### Pclass and fare are also highly inversely related as the fare of Pclass 1 would obviously be higher than corresponding Pclass 2 and 3 .
# ### Also , people with lower ages or children are travelling with their sibling and parents more than higher aged people (following an inverse relation) , which is quite a bit obvious .
# ### Parch and SibSp are also highly directly related
# ### Sex_bool and Survived people are highly inversely related , i.e. females are more likely to survive than men

# In[ ]:


df.groupby("Pclass").Age.mean()


# In[ ]:


df.isnull().sum()


# In[ ]:


for x in [dftrain, dftest,df]:
    x['Age_bin']=np.nan
    for i in range(8,0,-1):
        x.loc[ x['Age'] <= i*10, 'Age_bin'] = i


# In[ ]:


df[['Age','Age_bin']].head(20)


# In[ ]:


plt.figure(figsize=(20,20))
sns.set(font_scale=1)
sns.factorplot('Age_bin','Survived', col='Pclass' , row = 'Sex',kind="bar", data=df)


# In[ ]:


df.describe()


# # 25% QUARTILE IS 7 THEN 50 IS 14 , 75 IS 31 MAX 512 

# In[ ]:


for x in [dftrain, dftest , df]:
    x['Fare_bin']=np.nan
    for i in range(12,0,-1):
        x.loc[ df['Fare'] <= i*50, 'Fare_bin'] = i


# In[ ]:


fig, axes = plt.subplots(2,1)
fig.set_size_inches(20, 18)
sns.kdeplot(df.Age_bin , shade=True, color="red" , ax= axes[0])
sns.kdeplot(df.Fare , shade=True, color="red" , ax= axes[1])


# ### most passengers were from 20 to 40 
# ### most passengers paid nearly 40 rupees , one thing i can is that in case of fare we are facing left skewed graph need of coversion to log_scale or sqrt scale is necessary depending upon what box-cox transform value come up

# ### Now we go for Feature Engg finally as we have far much analysis and intitution is coming a use polynomial features too like
# ### features like Pclass with fare -> survived and many more combos with another
# 

# # data cleanning jobs are pending 1> filling of null values , imputaion 3> oulier cleaning

# In[ ]:


df.isnull().sum()


# In[ ]:




model= ols('Age~ Pclass + Survived + SibSp',data=df).fit()
print(model.summary())


# ### first filling in train set 1> embarked 2> then age and after that i will create a new varibale family size let's see what this do

# # now appending both the test and train frames first 

# In[ ]:


dftrain.info()


# In[ ]:


dftest.info()


# In[ ]:


np.where(dftrain["Embarked"].isnull())[0]


# In[ ]:


sns.factorplot(x='Embarked',y='Fare', hue='Pclass', kind="box",order=['C', 'Q', 'S'],data=dftrain, size=7,aspect=2)

# ... and median fare
plt.axhline(y=80, color='r', ls='--')


# # so we can see from red line that those who embarked from C HAS PAY 80 NEARLY , i was droppin these values , but now now 

# In[ ]:


dftrain.loc[[61,829],"Embarked"] = 'C'


# In[ ]:


dftrain.info()


# # now with dealing with age values 

# In[ ]:


fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
axis1.set_title('Original Age values - Titanic')
axis2.set_title('New Age values - Titanic')

# plot original Age values
# NOTE: drop all null values, and convert to int
dftrain['Age'].dropna().astype(int).hist(bins=70, ax=axis1)

# get average, std, and number of NaN values
average_age = dftrain["Age"].mean()
std_age = dftrain["Age"].std()
count_nan_age = dftrain["Age"].isnull().sum()

# generate random numbers between (mean - std) & (mean + std)
rand_age = np.random.randint(average_age - std_age, average_age + std_age, size = count_nan_age)

# fill NaN values in Age column with random values generated
age_slice = dftrain["Age"].copy()
age_slice[np.isnan(age_slice)] = rand_age

# plot imputed Age values
age_slice.astype(int).hist(bins=70, ax=axis2)


# In[ ]:


dftrain["Age"] = age_slice


# In[ ]:


dftrain.info()


# In[ ]:


dftrain=dftrain.drop('Age_bin',axis=1)


# In[ ]:


dftrain.info()


# In[ ]:


fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
axis1.set_title('Original Age values - Titanic')
axis2.set_title('New Age values - Titanic')

# plot original Age values
# NOTE: drop all null values, and convert to int
dftest['Age'].dropna().astype(int).hist(bins=70, ax=axis1)

# get average, std, and number of NaN values
average_age = dftest["Age"].mean()
std_age = dftest["Age"].std()
count_nan_age = dftest["Age"].isnull().sum()

# generate random numbers between (mean - std) & (mean + std)
rand_age = np.random.randint(average_age - std_age, average_age + std_age, size = count_nan_age)

# fill NaN values in Age column with random values generated
age_slice = dftest["Age"].copy()
age_slice[np.isnan(age_slice)] = rand_age

# plot imputed Age values
age_slice.astype(int).hist(bins=70, ax=axis2)


# In[ ]:


dftest["Age"] = age_slice


# # now both frames are filled up now we are left with cabin 

# In[ ]:


dftest.info()


# In[ ]:


dftest.info()


# In[ ]:


plt.figure(figsize=(20,20))
sns.factorplot(x='Fare',y='Cabin',data=dftrain,size=20)


# # as here we have stop because each cabin has unique but very since , each cabin were alloted in different way , unique things , and to make bins we need titanic full map , ship architecture

# In[ ]:


family_df = dftrain.loc[:,["Parch", "SibSp", "Survived"]]

# Create a family size variable including the passenger themselves
family_df["Fsize"] = family_df.SibSp + family_df.Parch + 1

family_df.head()


# In[ ]:


plt.figure(figsize=(15,5))

# visualize the relationship between family size & survival
sns.countplot(x='Fsize', hue="Survived", data=family_df)


# # as we can family with greater has less chance of survival 

# In[ ]:


dftrain['Fsize']=family_df['Fsize']


# In[ ]:


dftrain.info()


# In[ ]:


family_df_t= dftest.loc[:,["Parch", "SibSp", "Survived"]]

# Create a family size variable including the passenger themselves
family_df_t["Fsize"] = family_df_t.SibSp + family_df_t.Parch + 1

family_df_t.head()


# In[ ]:


dftest['Fsize']=family_df_t['Fsize']


# In[ ]:


dftest.info()


# In[ ]:


#dftest=dftest.drop('Cabin',axis=1)


# In[ ]:


dftest.info()


# In[ ]:


np.where(dftest["Fare"].isnull())[0]


# In[ ]:


dftest.ix[[152]]


# In[ ]:


dftest.loc[[152],"Fare"] = 10


# In[ ]:


dftest.ix[[152]]


# In[ ]:


dftest.info()


# In[ ]:


dftrain.info()


# In[ ]:


family_df_tr= dftrain.loc[:,["Parch", "SibSp", "Survived"]]

# Create a family size variable including the passenger themselves
family_df_tr["Fsize"] = family_df_tr.SibSp + family_df_tr.Parch + 1

family_df_tr.head()


# In[ ]:


dftrain['Fsize']=family_df_tr['Fsize']


# In[ ]:


dftrain['Fsize'].dtype


# In[ ]:


dftrain.info()


# In[ ]:


dftest.info()


# In[ ]:


import scipy.stats as stats
from scipy.stats import chi2_contingency

class ChiSquare:
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None #P-Value
        self.chi2 = None #Chi Test Statistic
        self.dof = None
        
        self.dfObserved = None
        self.dfExpected = None
        
    def _print_chisquare_result(self, colX, alpha):
        result = ""
        if self.p<alpha:
            result="{0} is IMPORTANT for Prediction".format(colX)
        else:
            result="{0} is NOT an important predictor. (Discard {0} from model)".format(colX)

        print(result)
        
    def TestIndependence(self,colX,colY, alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)
        
        self.dfObserved = pd.crosstab(Y,X) 
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof 
        
        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)
        
        self._print_chisquare_result(colX,alpha)

#Initialize ChiSquare Class
cT = ChiSquare(dftrain)

#Feature Selection
testColumns = ['Embarked','Cabin','Pclass','Age','Name','Fare','Fare_bin','Fsize']
for var in testColumns:
    cT.TestIndependence(colX=var,colY="Survived" )  


# In[ ]:


# Make a copy of the titanic data frame
dftrain['Title'] = dftrain['Name']

# Grab title from passenger names
dftrain["Title"].replace(to_replace='(.*, )|(\\..*)', value='', inplace=True, regex=True)


# In[ ]:


rare_titles = ['Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']
dftrain['Title'].replace(rare_titles, "Rare title", inplace=True)

# Also reassign mlle, ms, and mme accordingly
dftrain['Title'].replace(["Mlle","Ms", "Mme"], ["Miss", "Miss", "Mrs"], inplace=True)


# In[ ]:


dftrain.info()


# In[ ]:


cT = ChiSquare(dftrain)

#Feature Selection
testColumns = ['Embarked','Cabin','Pclass','Age','Name','Fare','Fare_bin','Fsize','Title','SibSp','Parch']
for var in testColumns:
    cT.TestIndependence(colX=var,colY="Survived" )  


# In[ ]:


dftest.info()


# In[ ]:


dftrain.info()


# In[ ]:


dftest=dftest.drop(['Ticket','PassengerId'],axis=1)


# In[ ]:


dftest.info()


# In[ ]:


dftest['Title'] = dftest['Name']

# Grab title from passenger names
dftest["Title"].replace(to_replace='(.*, )|(\\..*)', value='', inplace=True, regex=True)


# In[ ]:


dftest.info()


# In[ ]:


dftrain.info()


# In[ ]:


dftrain.head()


# ### cleanning the dftrain

# In[ ]:


dftrain=dftrain.drop('Name',axis=1)


# In[ ]:


dftrain.head()


# In[ ]:


context1 = {"female":0 , "male":1}
context2 = {"S":0 , "C":1 , "Q":2}
dftrain['Sex_bool']=dftrain.Sex.map(context1)
dftrain["Embarked_bool"] = dftrain.Embarked.map(context2)


# In[ ]:


dftrain.head()


# In[ ]:


#dftrain=dftrain.drop(['Sex','Embarked'],axis=1)
context3= {"Mr":0 , "Mrs":1 , "Miss":2,'Master':3}
dftrain['Title']=dftrain.Title.map(context3)


# In[ ]:


dftrain.head()


# In[ ]:


dftrain=dftrain.drop(['PassengerId','Cabin','Ticket'],axis=1)
plt.figure(figsize=(14,4))
sns.boxplot(data=dftrain)


# In[ ]:


reserve=dftrain.copy()
reserve.shape


# In[ ]:


dftrain.head()


# In[ ]:


dftrain=dftrain.drop(['Embarked','Sex'],axis=1)
dftrain.head()


# In[ ]:


#dftrain=dftrain[np.abs(zscore(dftrain)<3).all(axis=1)]


# In[ ]:


dftest.head()


# In[ ]:


context1 = {"female":0 , "male":1}
context2 = {"S":0 , "C":1 , "Q":2}
dftest['Sex_bool']=dftest.Sex.map(context1)
dftest["Embarked_bool"] = dftest.Embarked.map(context2)
context3= {"Mr":0 , "Mrs":1 , "Miss":2,'Master':3}
dftest['Title']=dftest.Title.map(context3)


# In[ ]:


dftest.head()


# In[ ]:


dftest=dftest.drop(['Name','Sex','Embarked'],axis=1)


# In[ ]:


dftest.head()


# In[ ]:


for x in [dftrain, dftest,df]:
    x['Age_bin']=np.nan
    for i in range(8,0,-1):
        x.loc[ x['Age'] <= i*10, 'Age_bin'] = i


# In[ ]:


dftrain.head()


# In[ ]:


dftest.head()


# In[ ]:


dftrain=dftrain.drop(['Fare_bin'],axis=1)
dftest=dftest.drop(['Fare_bin','Cabin'],axis=1)
for x in [dftrain, dftest,df]:
    x['Fare_bin']=np.nan
    for i in range(12,0,-1):
        x.loc[ x['Fare'] <= i*10, 'Fare_bin'] = i


# In[ ]:


dftrain.head()


# In[ ]:


dftest.head()


# In[ ]:


dftrain=dftrain.drop(['Age','Fare'],axis=1)
dftest=dftest.drop(['Age','Fare'],axis=1)


# In[ ]:


dftrain.head()
dftrain=dftrain.convert_objects(convert_numeric=True)


# In[ ]:


def change_type(df):
    float_list=list(df.select_dtypes(include=["float"]).columns)
    print(float_list)
    for col in float_list:
        df[col]=df[col].fillna(0).astype(np.int64)
        
    return df    
change_type(dftrain)    
dftrain.dtypes


# In[ ]:


#dftrain=dftrain.drop(['Fare'],axis=1)
#dftest=dftest.drop(['Fare','Cabin'],axis=1)
x=dftrain.iloc[:,1:].values
y=dftrain.iloc[:,0].values
print(dftrain.columns)
print(dftest.columns)

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=101)


# In[ ]:


dftest=dftest.convert_objects(convert_numeric=True)
change_type(dftest)    
dftest.dtypes


# In[ ]:


MLA = []
Z = [LinearSVC() , DecisionTreeClassifier() , LogisticRegression() , KNeighborsClassifier() , GaussianNB() ,
    RandomForestClassifier() , GradientBoostingClassifier()]
X = ["LinearSVC" , "DecisionTreeClassifier" , "LogisticRegression" , "KNeighborsClassifier" , "GaussianNB" ,
    "RandomForestClassifier" , "GradientBoostingClassifier"]

for i in range(0,len(Z)):
    model = Z[i]
    model.fit( X_train , y_train )
    pred = model.predict(X_test)
    MLA.append(accuracy_score(pred , y_test))


# In[ ]:


MLA


# In[ ]:


sns.kdeplot(MLA , shade=True, color="red")


# In[ ]:


d = { "Accuracy" : MLA , "Algorithm" : X }
dfm = pd.DataFrame(d)


# In[ ]:


dfm


# In[ ]:


sns.barplot(x="Accuracy", y="Algorithm", data=dfm)


# In[ ]:


# imporvsing the model first logistic Regression
params={'C':[1,100,0.01,0.1,1000],'penalty':['l2','l1']}
logreg=LogisticRegression()
gscv=GridSearchCV(logreg,param_grid=params,cv=10)
get_ipython().magic(u'timeit gscv.fit(x,y)')


# In[ ]:


gscv.best_params_


# In[ ]:


logregscore=gscv.best_score_
print(logregscore)


# In[ ]:


gscv.predict(X_test)
gscv.score(X_test,y_test)


# # logistic Regressor increased by 1%

# In[ ]:


rfcv=RandomForestClassifier(n_estimators=500,max_depth=6)
rfcv.fit(X_train,y_train)
rfcv.predict(X_test)
rfcv.score(X_test,y_test)


# # Random forest accuracy increased to 82.4%

# In[ ]:


gbcv=GradientBoostingClassifier(learning_rate=0.001,n_estimators=2000,max_depth=5)
gbcv.fit(X_train,y_train)
gbcv.predict(X_test)
gbcv.score(X_test,y_test)


# # Gradient Boosting Classifier increased to 1.3% accuracy

# # Now going for KNN

# In[ ]:


param={'n_neighbors':[3,4,5,6,8,9,10],'metric':['euclidean','manhattan','chebyshev','minkowski'] }       
knn = KNeighborsClassifier()
gsknn=GridSearchCV(knn,param_grid=param,cv=10)
gsknn.fit(x,y)                         
                                                


# In[ ]:


gsknn.best_params_


# In[ ]:


gsknn.best_score_


# In[ ]:


gsknn.predict(X_test)


# In[ ]:


gsknn.score(X_test,y_test)


# ## KNN With neigbhour = 5 and metric = euclidean  and accuracy score is 85.074 and it is cross validated 

# In[ ]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, gscv.predict_proba(X_test)[:,1])
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rfcv.predict_proba(X_test)[:,1])
knn_fpr, knn_tpr, knn_thresholds = roc_curve(y_test, gsknn.predict_proba(X_test)[:,1])
gbc_fpr, gbc_tpr, ada_thresholds = roc_curve(y_test, gbcv.predict_proba(X_test)[:,1])

plt.figure(figsize=(9,9))
log_roc_auc = roc_auc_score(y_test, gscv.predict(X_test))
print ("logreg model AUC = {} " .format(log_roc_auc))
rf_roc_auc = roc_auc_score(y_test, rfcv.predict(X_test))
print ("random forest model AUC ={}" .format(rf_roc_auc))
knn_roc_auc = roc_auc_score(y_test, gsknn.predict(X_test))
print ("KNN model AUC = {}" .format(knn_roc_auc))
gbc_roc_auc = roc_auc_score(y_test, gbcv.predict(X_test))
print ("GBC Boost model AUC = {}" .format(gbc_roc_auc))
# Plot Logistic Regression ROC
plt.plot(fpr, tpr, label='Logistic Regression')

# Plot Random Forest ROC
plt.plot(rf_fpr, rf_tpr, label='Random Forest')

# Plot Decision Tree ROC
plt.plot(knn_fpr, knn_tpr, label=' KnnClassifier')

# Plot GradientBooseting Boost ROC
plt.plot(gbc_fpr, gbc_tpr, label='GradientBoostingclassifier')

# Plot Base Rate ROC
plt.plot([0,1], [0,1],label='Base Rate' 'k--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




