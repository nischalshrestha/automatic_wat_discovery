#!/usr/bin/env python
# coding: utf-8

# **Hi, I am Simha. I am new to Python and currently learning the new stuff.! Firstly I have imported the libraries and loaded the data. Next I have explored the data sets and identified the missing values. I have next imputed the missing values and started performing modelling. I have compared different models and picked the best one with highest score. Thanks to the work of other Kaggle users, I used it as a reference and started exploring it in my way. Feeling more positive after handling this data set. !** 

# In[ ]:


# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
titanic = pd.read_csv("../input/train.csv")

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

titanic = pd.read_csv("../input/train.csv")
titanic.head(50)  #displays 50 rows


# In[ ]:


round(titanic.describe(),2) #rounding off the values to two integers
# There are some missing valuees in Age
# We know the mean age is 29.70 from the values we have


# In[ ]:


titanic.isnull().sum()
#We are finding out the total number of missing values in Age and other variables.
#From the below output we can understand there are missing values in Cabin, 
# and Embarked variable in addition to Age(which we already know)
# We can also use titanic.info() and find out the same.(if the value is less than 891 then it has missing value)


# In[ ]:


titanic_test=pd.read_csv("../input/test.csv")
titanic_test.isnull().sum()
#There are missing values in Age and Cabin in test data set


# In[ ]:


titanic.isnull().sum() - titanic_test.isnull().sum()

#Now we know there is no survived variable in test data set


# In[ ]:


#Visualization
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=0.7)
pd.options.display.mpl_style = 'default'
titanic.hist(bins=10,figsize=(9,8),grid=False)
# x axis had count and y axis values are split based on bins


# In[ ]:


g1 = sns.FacetGrid(titanic, col="Sex", row="Survived", margin_titles=True)
g1.map(plt.hist,"Age",color="red")


# In[ ]:


g2 = sns.FacetGrid(titanic, hue="Survived", col="Pclass", margin_titles=True,
               palette={1:"red", 0:"green"})
g2=g2.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend()


# In[ ]:


titanic.Embarked.value_counts().plot(kind='bar',color="violet", alpha=0.9)
#alpha is the tranperency of the cell (for a lighter purple color choose low alpha 
#                                      and for darker vice versa)
plt.title("Passengers boarding from")
#(C = Cherbourg; Q = Queenstown; S = Southampton)
# better way is to represent this graph in a pie diagram


# In[ ]:


sns.factorplot(x = 'Embarked',y= "Survived", data = titanic, color="b")
# C = Cherbourg; Q = Queenstown; S = Southampton
# The survival"pr", bability of people who have boarded from Cherbourg is higher than the other two
            # (Correct me if i have interpreted the graph wrongly)
            # factor graph is a really great area to learn. I have read many articles on factoring after this graph


# In[ ]:


#This graph helps us to know the survival rate of men and women with reference to each subclass
sns.set(font_scale=1)
g3 = sns.factorplot(x="Sex", y="Survived", col="Pclass",
                    data=titanic, saturation=0.8,
                    kind="bar", ci=None, aspect=.6)
# saturation determines the color of the cell and aspect decides the size of the cell
(g3.set_axis_labels("", "Survival Rate")
    .set_xticklabels(["Men", "Women"])
    .set_titles("{col_name} {col_var}")
    .set(ylim=(0, 1))
    .despine(left=True))
# Xtick lables are the name of the bar's inside each subclass. i.e. men and women
# ylim varaibles tell the range of the values on y axis (0 to 1)
plt.subplots_adjust(top=0.8)
# subplots_adjust(top makes us to adjust the height of the bar)
g3.fig.suptitle('Survival rate of men and women in each subclass')


# In[ ]:


ax = sns.boxplot(x="Survived", y="Age", data=titanic, saturation=0.8)
# the results are interesting. Survival rate of people within 60 years of age is more
# other words, people who are younger had a higher chance of survival
# But the median value of people who have survived and not usrvived are close to 28 years of age
# Mean always averages the age value but Median picks the middle value. The best in this instance.


# In[ ]:


# Now we will draw an line graph for survived and non survived people

g4 = sns.FacetGrid(titanic, hue="Survived",aspect=2.5)
g4.map(sns.kdeplot,'Age',shade= True)
g4.set(xlim=(0, titanic['Age'].max()))
g4.add_legend()

# Kids between 0 - 12 years of age and Adults who are between 30 - 38 years of age had 
#                                   a higher chance of survival.


# In[ ]:


# Different way of bringing in three graphs into one graph

titanic.Age[titanic.Pclass == 1].plot(kind='kde')    
titanic.Age[titanic.Pclass == 2].plot(kind='kde')
titanic.Age[titanic.Pclass == 3].plot(kind='kde')
plt.xlabel("Age")    # x axis label
plt.title("Age vs.Passenger class") #legend
plt.legend(('Class 1', 'Class 2','Class 3'),loc='best') 


# In[ ]:


corr=titanic.corr() #(correlation of all variables in your data set)
plt.figure(figsize=(8,8)) #let have a square shaped output
sns.heatmap(corr, vmax=1, square=True,annot=True,cmap='cubehelix')
plt.title('Correlation b/w all values')

# from the below output the maximum correlation was between Parch and Sibsp
# sibsp           Number of Siblings/Spouses Aboard
# parch           Number of Parents/Children Aboard

# In Titanic, there were more families with sibilings, spouses, parents and children on board


# Intestingly passenger class and fare are negatively correlated.
# Lets find out later why passenger class and fare are negatively correlated as 
#                                                 our focus is more on survival

# to display values with reference to survived
round(titanic.corr()["Survived"],2)
# the output shows that fare had a positive correlation with survival rate
# also, passenger class is negatively correlated with survival rate


# In[ ]:


round(titanic.corr()["Survived"],2).plot(kind='bar', color="blue", alpha=0.9)
# the output shows that fare had a positive correlation with survival rate
# also, passenger class is negatively correlated with survival rate


# In[ ]:


g6 = sns.factorplot(x="Age", y="Embarked", hue="Sex", row="Pclass",
                    data=titanic[titanic.Embarked.notnull()],orient="h", size=2, aspect=3.5, 
                   palette={'male':"red", 'female':"green"}, kind="violin", split=True, cut=0, bw=.2)


# In[ ]:





# In[ ]:


round(titanic.corr()["Survived"],2).plot(kind='bar', color="blue", alpha=0.9)
# the output shows that fare had a positive correlation with survival rate
# also, passenger class is negatively correlated with survival rate


# In[ ]:


g6 = sns.factorplot(x="Age", y="Embarked", hue="Sex", row="Pclass",
                    data=titanic[titanic.Embarked.notnull()],orient="h", size=2, aspect=3.5, 
                   palette={'male':"red", 'female':"green"}, kind="violin", split=True, cut=0, bw=.2)


# In[ ]:


def plot_model_var_imp( model , X , y ):
    imp = pd.DataFrame( 
        model.feature_importances_  , 
        columns = [ 'Importance' ] , 
        index = X.columns 
    )
    imp = imp.sort_values( [ 'Importance' ] , ascending = True )
    imp[ : 10 ].plot( kind = 'barh' )
    print (model.score( X , y ))


# In[ ]:


Missing Values Imputation


# In[ ]:


full = titanic.append(titanic_test, ignore_index= True)
print ('Full data set',full.shape, 'titanic:',titanic.shape)
# we are taking appending full data set to one place so that it will be easy to impute missing values


# In[ ]:


impute = pd.DataFrame()
impute ['Age']= full.Age.fillna(full.Age.mean())
impute ['Fare']= full.Age.fillna(full.Age.mean())
impute.isnull().sum()


# In[ ]:


full.isnull().sum()
#See the difference.! Impute carries new values 


# In[ ]:


# Cabin stil has missing values. We can replace the variables with U


# In[ ]:


cabin = pd.DataFrame()
cabin['Cabin' ] = full.Cabin.fillna ('U')
cabin['Cabin'] = cabin['Cabin'].map(lambda c: c[0])
cabin = pd.get_dummies(cabin['Cabin'], prefix = 'Cabin')
cabin.head()


# In[ ]:


title = pd.DataFrame()
title[ 'Title' ] = full[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )
Title_Dictionary = {
                    "Capt":       "High Designation",
                    "Col":        "High Designation",
                    "Major":      "High Designation",
                    "Jonkheer":   "High Status",
                    "Don":        "High Status",
                    "Sir" :       "High Status",
                    "Dr":         "High Designation",
                    "Rev":        "High Designation",
                    "the Countess":"High Designation",
                    "Dona":       "High Designation",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Mr",
                    "Lady" :      "High Designation"
                    }
title[ 'Title' ] = title.Title.map( Title_Dictionary )
title = pd.get_dummies( title.Title )
title.head()


# In[ ]:


family = pd.DataFrame()
family[ 'FamilySize' ] = full[ 'Parch' ] + full[ 'SibSp' ] + 1
family[ 'Family_Single' ] = family[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
family[ 'Family_Small' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 3 else 0 )
family[ 'Family_Large' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 4 <= s else 0 )
family.head()


# In[ ]:


age1 = pd.DataFrame()
age1[ 'AgeGroup' ] = impute[ 'Age' ]
age1[ 'Until10' ] = age1[ 'AgeGroup' ].map( lambda r : 1 if r <= 10 else 0 )
age1[ '11to25' ] = age1[ 'AgeGroup' ].map(  lambda r : 1 if 11 <= r <= 25 else 0 )
age1[ '26to40' ]  = age1[ 'AgeGroup' ].map( lambda r : 1 if 26 <= r <= 40 else 0 )
age1[ '40to60' ]  = age1[ 'AgeGroup' ].map( lambda r : 1 if 41 <= r <= 60 else 0 )
age1[ '60Plus' ]  = age1[ 'AgeGroup' ].map( lambda r : 1 if 61 <= r else 0 )
age1.head(50)


# In[ ]:


full_x = pd.concat((impute,cabin,age1,family,title), axis=1)
full_x.head()
# concating new variables after f


# In[ ]:


train_valid_x = full_x[0:891]
train_valid_y = titanic.Survived
test_x = full_x[891:]
train_x, valid_x, train_y, valid_y = train_test_split(train_valid_x, train_valid_y, train_size=0.7)
print(full_x.shape,train_x.shape, valid_x.shape, train_y.shape, valid_y.shape, test_x.shape)

#SPLITTING INTO TRAINING AND TESTING DATASETS


# In[ ]:


#Modelling


# In[ ]:


#Random Forest Model
model1 = RandomForestClassifier(n_estimators=100)
#support vector machines
model2 = SVC()
#gradient boosting classifier
model3 = GradientBoostingClassifier()
#K nearest neighbour
model4 = KNeighborsClassifier(n_neighbors = 3)
#gaussian naive bayes
model5 = GaussianNB()
#Logistic Regression
model6 = LogisticRegression()


# In[ ]:


#Training the model
model1.fit(train_x,train_y)
model2.fit(train_x,train_y)
model3.fit(train_x,train_y)
model4.fit(train_x,train_y)
model5.fit(train_x,train_y)
model6.fit(train_x,train_y)


# In[ ]:


#Model Performance
plot_model_var_imp(model1, train_x, train_y)


# In[ ]:


#Model Performance
print ('Model 2', model2.score( train_x , train_y ) , model2.score( valid_x , valid_y ))
print ('Model 3', model3.score( train_x , train_y ) , model3.score( valid_x , valid_y ))
print ('Model 4', model4.score( train_x , train_y ) , model4.score( valid_x , valid_y ))
print ('Model 5', model5.score( train_x , train_y ) , model5.score( valid_x , valid_y ))
print ('Model 6', model6.score( train_x , train_y ) , model6.score( valid_x , valid_y ))


# In[ ]:


rfecv = RFECV( estimator = model1, step = 1 , cv = StratifiedKFold( train_y , 2 ) , scoring = 'accuracy' )
rfecv.fit( train_x , train_y )


# In[ ]:


#completion time


# In[ ]:


test_y = model1.predict( test_x )
passenger_id = full[891:].PassengerId
test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_y } )
test.shape
test.head()
test.to_csv( 'titanic_prediction2.csv' , index = False )

