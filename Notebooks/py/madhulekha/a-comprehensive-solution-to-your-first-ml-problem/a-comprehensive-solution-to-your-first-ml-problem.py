#!/usr/bin/env python
# coding: utf-8

# Hi everyone, 
# 
# This is my first kernel and am writing this with an intention of sharing  the journey to my first ML and Kaggle problem. The accuracy can still be improved and will work on it soon. :) 
# 
# I came up to this problem straight away after Andrew NG's Introduction to Machine Learning on Course Era. A pretty standard route for people trying to start of with ML and a decent one too. In this notebook, I tried to jot down the steps in building a predictive model for Titanic and some resources alongside. If its your first problem, its important that you take pauses in between, ponder over what you've done till then, read a bit about the step and proceed (Don't just read - read and try that segment - Its an iterative process). Having said this, there's information explosion around and its very easy to get lost (I know I'm adding more!). Its very essential that you find 2-3 kernels or notebooks which are understandable by you and stick to them.   
# 
# This model achieves a score of 0.7894, which is in the top 5% of all submissions at the time of this writing. Now that I've reached here (from 0.72248), I finally can understand and appreciate why this is a great introductory modeling exercise. The nature of the data is simple - with decent number of variables, scope for nice visualizations and can instill a sense of systematic approach that's to be taken for solving such problems. he process goes something like this - 
# 
# **1.** **Understand the variables (Exploratory Data Analysis):**  
#      a) Look at each individual variable and it's distribution/ counts <br>
#      b) Look at relation between the target (Survival) and remaining variables in detail <br>
#      c) We'll try to understand how the dependent variables relate amongst themselves and combinedly impact targer
#      
#    Here, we can club 'a' and 'b' steps as the number of variables are less and almost all of them are discrete. <br>
#    
#    Seaborn now becomes our alter ego. 
#    [Official Documentation](http://seaborn.pydata.org/tutorial.html )of seaborn is quite elaborate and neat. It might seem a bit of a      patience tester, but worth the go through if you've time and especially if you are more of a data analyst than a scientist. A              bottom-up approach would be to look at the [example gallery](http://seaborn.pydata.org/examples/index.html), pick the graph        you want to represent your variables with and dive in. Another shortcut is datacamp's [Ultimate Python Seaborn Tutorial](http://datacamp.com/community/open-courses/kaggle-python-tutorial-on-machine-learning#gs.=IXCXgA)
#    
#  **2.** **Data Cleaning :** Impute the missing values and handle categorical variables (In regression problems, we can think of outliers too)
#  
#  **3. Feature Engineering :** Taking learnings from 1, we'll create new variables which might capture the trends we see as important and the algorithm might otherwise miss out. This particular problem requires basic feature engineering - [Here](https://elitedatascience.com/feature-engineering-best-practices) is a nice article I found. There's hell load of literature around this especially in image and speech recognition fields.
#  
#  **4. Modeling :**  I've used python before, but not for modeling, so wandered a bit around on the how-to (Andrew teaches it on Matlab and he gets us to implement the math of an algorithm - which is not the case here. Using ML is much much simpler than that :P). The [intro to scikit learn by Kevin Markham](http://www.dataschool.io/machine-learning-with-scikit-learn/ ) is must do for beginners. Its relatively slow paced and gets too basic at times. But I think everyone should watch it (even at 2x). Error analysis is an integral part of modeling. So we'll fit a model, cross validate it and tune hyper parameters (Watching the dataschool video above should clear this part)
#  
#  Let's get started !
#  ***

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #Visualization
import matplotlib.pyplot as plt #Visualization
get_ipython().magic(u'matplotlib inline')
# ^^ Will make your plot outputs appear and be stored within the notebook.

from itertools import chain #For ironing out lists - Can be avoided. 
                            #Using it as it'll be useful in Python for Data analyses in general

#Classifiers
from sklearn.ensemble import RandomForestClassifier
    


# Try to keep libraries in a different cell as we would not want to run them everytime. Reading below the input test and train csvs and checking their size to make sure they're loaded correctly.

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print("Train set size:",train.shape,"|| Test set size:",test.shape)


# # **EXPLORATORY DATA ANALYSIS**<br>
# ##  **Train Set Overview**<br>
# Let's have a first look of data in train set i.e columns, their datatypes and nulls. <br>
# Age, Cabin and Embarked have null values.

# In[ ]:


train.info()


# In[ ]:


train.head(3)


# A little more description about the stats of the variables overall before moving on to visualisations. 
# 
#  - Pasenger ID is a normal index variable, so we can discard it. I'm going to keep it till the end, just to be able to keep track of rows while merging test and train (Which I'll do during data cleaning)  
#  - Average age is ~30 with 25-50-75 percentiles also showing values from 20-38 (Though the max and min are 0.4 and 80). Intereresting !
#  - Fare seems to have a distribution to look at with quantiles nicely dispersed.  
#  - Ticket has 681/891 uniques. So there seems to me some value we can extract by grouping the variable or something of sorts from this seemingly junky variable. 

# In[ ]:


train.describe()#Numerical Variables


# In[ ]:


train.describe(include=['O'])  #Categorical Variables


#  ## **Individual Variables Overview**

# ****1-5 : Categorical/ Ordinal Features****
# 1. PClass : Clearly 1st class has higher survivial. 
# 2. Sex : Women though are less have higher survivial rate. 
# 3. SibSp and Parch : Survivial decreased drastically for large families. Mid-sized families fared well. People alone are high in number and have low survivial rates too. So we can create a new variable adding these two.
# 4. Embarked : Embarked C has a higher survival rate. Might be because they're closer to escape or might be becase most of the 1st class is accommodated there. 

# In[ ]:


categ_vars = ['Survived','Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
fig, ax = plt.subplots(nrows = 2, ncols = 3 ,figsize=(20,10))
fig.subplots_adjust(wspace=0.4, hspace=0.4)
ax = list(chain.from_iterable(ax)) #Change ax from matrix to a list for iteration 
for i in range(len(categ_vars)):
    sns.countplot(train[categ_vars[i]], hue=train['Survived'], ax=ax[i])


# **6 : Name**
# 
# The name column as such can't be used. We know that not all datapoints will be in ideal forms for us to use directly. So a quick run through of names will bring the 'title' aspect to our notice. We may be able to categorize our data based on that. 

# In[ ]:


train['Title'] = train['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
tab = pd.crosstab(train['Title'],train['Survived'])
print(tab)
tab_prop = tab.div(tab.sum(1).astype(float), axis=0)
tab_prop.plot(kind="bar", stacked=True)


# Higher namelengths showed higher survival rate than average. Important people/ royalty had longer names. So including that attribute

# In[ ]:


train['Name_Len'] = train['Name'].apply(lambda x: len(x))
print(train['Survived'].groupby(pd.qcut(train['Name_Len'],5)).mean(),
      pd.qcut(train['Name_Len'],5).value_counts())


# **7 : Cabin** <br>
# The variable has extreme amount of nulls, but by description seems something which can contribute a lot. The deck determines the proximity to escape routes. Ignoring the cabin number as of now. It doesn't seem to have to do anything with survivial. It just goes with the cabin letter.

# In[ ]:


train['Deck'] = train['Cabin'].apply(lambda x: str(x)[0])
sns.countplot(train['Deck'], hue=train['Survived'])


# **8 : Ticket** <br>
# In my first model I totally ignored this variable after reading some posts given its randomness. I was frustrated at a point about accuracy not increasing and by old habit loaded the data into excel just to see (As the size is less too..). I thought I could impute some of the missing cabin values based on ticket numbers so I split the alphanumeric values. That didn't perform much well - The alphabet variable is getting useless now as it's having missing values. For uniformity I took all the first characters of ticket which might be indicative of potential information similar to cabin. 
# 
# Struggling at ~80 LB score and looking for ideas to improve it, ended up with adding ticket len variable which helped jump up the LB. It seems reflective of the position of passenger on the deck similar to cabin. 

# In[ ]:


train['Ticket_Lett'] = train['Ticket'].apply(lambda x: str(x)[0])
sns.countplot(train['Ticket_Lett'], hue=train['Survived'])


# In[ ]:


train['Ticket_Len'] = train['Ticket'].apply(lambda x: len(x))
sns.countplot(train['Ticket_Len'], hue=train['Survived'])


# **9-10 : Age - Fare**<br>
# 
# For now, I'm going ahead with filling null values with -1 and see how the distribution of Age is. With fare I have a doubt whether the fare is per person or the whole family (Which is not clear in the variable description on kaggle, I saw no way to figure it out too). So I'm going ahead using it as is without trying much to explain the distribution. The one point seems to be an outlier in fare variable. Printing out the row to check. 

# In[ ]:


train['Age'].fillna(-1,inplace=True)
fig, ax = plt.subplots(nrows = 1, ncols = 2 ,figsize=(20,8))
age = sns.distplot(train['Age'].dropna(), label='Total',bins=12,kde =False,ax=ax[0])
age = sns.distplot(train[train['Survived']==1].Age.dropna(), label='Survived',bins=12,kde =False,ax=ax[0])
age.legend()

fare = sns.distplot(train['Fare'], label='Total',bins=12,kde =False,ax=ax[1])
fare = sns.distplot(train[train['Survived']==1].Fare, label='Survived',bins=12,kde =False,ax=ax[1])
fare.legend()


# In[ ]:


train.loc[train.Fare.argmax()]


#  ## **Relations between variables**

# A quick thing to do is pairplot as the number of features are manageable. Its amazing to see how seaborn concisely presents so much information at one go !
# 
# From this, Fare clearly has to do with Passenger Class, similar is ticket length (Looks like this is a significant variable). Other than that there is nothing much to guage, but that atleast leaves us with having to look at relationships between rest of variables.  
# i.e. Sex, Cabin, Embarked and Name.

# In[ ]:


#sns.pairplot(train[train.columns[train.columns!='Survived'] ])
sns.pairplot(train.drop(['Survived','PassengerId'],axis=1))


# **1 : Gender - Class - Survival**<br>
# Majority of the women who died are from 3rd class. Almost all women from other classes survived.

# In[ ]:


sns.factorplot(x="Pclass", hue="Survived", col="Sex",data=train, kind="count");


# **2 : Gender - Embarked - Survival**<br>
# Majority of the women who died are embarked S. But that's the general trend of embarked too. So we'll go ahead and see relation between embarked and class.

# In[ ]:


sns.factorplot(x="Embarked", hue="Survived", col="Sex", data=train, kind="count");


# **3 : Embarked - Class - Survival**<br>
# Embarked C has higher survival rate due to more Class one passengers in there. Embarked Q 3rd class has higher relative rate of survival as compared to other locations. One more thing to note is that there are very less 1st and 2nd class embarked Q.  //Need to come up with better way to visualize this ! :/ 

# In[ ]:


sns.factorplot(x="Embarked", hue="Pclass", col="Survived", data=train, kind="count");


# # **FEATURE ENGINEERING**<br>
# 
# We've done decent exploration of variables. It's not exhaustive. I've tinkered a lot around the variables. These are some observations which are worth mentioning. Lets move ahead now taking the ideas of what we want to build. 
# 
# For ease I'm going to club both train and test sets and apply any transformations together on both of them. 
# 
# High variance about the overall mean of target can be one metric for a good feature. It may help our model in classifying.
# 

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

predictor = train["Survived"]

all_data = (train.drop(["Survived"] , axis=1)).append(test);
all_data.isnull().sum()


# Imputing missing fare and embarked values with mean and mode. There are hardly any. 

# In[ ]:


all_data['Fare'].fillna(all_data['Fare'].mean(), inplace = True)
all_data['Embarked'].fillna('S', inplace = True) #all_data['Embarked'].mode() doesn't work because of NAs


# We'll combine the siblings and parents variables. We'll also imbibe our observations about midsize families in this variable by recategorizing it as alone, midsize and large families.  

# In[ ]:


all_data['FamSz'] = all_data['SibSp'] + all_data['Parch']
all_data['FamSz'] = np.where(all_data['FamSz'] == 0 , 'Alone',
                           np.where(all_data['FamSz'] <= 3,'Midsize', 'Large'))


# We'll consider class as a categorical variable, so that our algorithm will be able to capture the impact of class efficiently. Carrying out all other steps we decided on doing during data exploration.

# In[ ]:


#Name
all_data['Title'] = all_data['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
all_data['NameLen'] = all_data['Name'].apply(lambda x: len(x))

#Title
all_data['TicketLett'] = all_data['Ticket'].apply(lambda x: str(x)[0])
all_data['TicketLen'] = all_data['Ticket'].apply(lambda x: len(x))

#Cabin
all_data['Deck'] = all_data['Cabin'].apply(lambda x: str(x)[0])

#Class
all_data['Pclass'] = all_data['Pclass'].astype(str)

all_data.drop(['Cabin','Ticket','Name','SibSp','Parch','PassengerId'],axis=1,inplace=True)


# Imputing age values based on Class and Title instead of plain mean for more accuracy. I think we can use a random forest in itself here to get more accurate values of age basis Name_Title, Name_Len, Pclass and Fare

# In[ ]:


all_data['AgeNull'] = all_data['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)
data = all_data.groupby(['Title', 'Pclass'])['Age']
all_data['Age'] = data.transform(lambda x: x.fillna(x.mean()))
all_data['Age'].fillna(all_data['Age'].mean(),inplace=True)

print(all_data.isnull().sum()) #Sanity check that all values are filled


# #######<br>
# The above statement will bring us that one of the age values is na. Checkin for which row it is by using
# all_data[all_data['Age'] != all_data['Age']] 
# #######<br>
# This row is a catch as its the only entry after groupby. So adding another step in the cleaning of Age to impute a value into this observation.

# Random forest is biased towards categorical variables with more classes. To avoid that and in general also to prevent over fitting. I'm grouping title and ticket variables.

# In[ ]:


all_data['Title'] = np.where((all_data['Title']).isin(['Col.', 'Mlle.', 'Ms.','Major.','the','Lady.','Jonkheer.', 'Sir.',
                                    'Capt.','Don.','Dona.' ,'Mme.']), 'Rare',all_data['Title'])
all_data['TicketLett']= all_data['TicketLett'].apply(lambda x: str(x))
all_data['TicketLett'] = np.where((all_data['TicketLett']).isin(['W', '7', 'F','4', '6', 'L', '5', '8','9']), 'Rare',
                                            all_data['TicketLett'])


# Encode all categorical values. Pandas again has this amazing function dummies for handling these.

# In[ ]:


for_encoding = list(all_data.select_dtypes(include=["object"]))
remaining_cols = list(all_data.select_dtypes(exclude=["object"]))
numerical = all_data[remaining_cols]
encoded = pd.get_dummies(all_data[for_encoding])
all_data_new = pd.concat([numerical,encoded],axis=1)
print(len(all_data_new.columns))
train_new = all_data_new[0:len(train)]
test_new = all_data_new[len(train)::]
print(train_new.shape,test_new.shape)
print(all_data_new.columns)
print(train_new.dtypes)


# # **MODELLING**<br>
# 
# I've tried out different models logistic regression, knn, basic neural networks. Didn't maintain proper versions + Random forest performed the best as expected. Reporting only the final model. 
# 
# Hyperparameter tuning is extremely important for extracting the maximum out of the model. As any beginner, I underestimated the capacity of this. I tried random forest initially after the feature engineering and was lingering around 0.74 LB Score. Started to look at more complicated models, ensembles and what not. Came back to random forest after [reading this](https://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/). Kevin Markham's videos must have introduced you to GridSearch. It saves us the effort of writing for loops to iterate over multiple parameters and of course is highly optimized. 
# 
# Running the model for optimal parameters (I've run it on my local system, to skip the load time here. 
# 

# In[ ]:


rf1 = RandomForestClassifier(criterion= 'gini',
                             n_estimators=100,
                             min_samples_split=4,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)
rf1.fit(train_new, predictor)
print("%.4f" % rf1.oob_score_)


# In[ ]:


importances = pd.DataFrame({'feature':train_new.columns,'importance':np.round(rf1.feature_importances_,3)})
importances = importances.sort_values('importance', ascending=False).set_index('feature')
importances[0:15].plot.bar()


# In[ ]:


predictions = rf1.predict(test_new)
output = pd.DataFrame({ 'PassengerId': range(892,1310),
                            'Survived': predictions  })
output.to_csv('submission_madhu_rf.csv',index_label=False,index=False)


# ***
# Hope this helped :). Cheers ! 
# 
# P. S. : If you want to make your own kernel, [this article](https://medium.com/ibm-data-science-experience/markdown-for-jupyter-notebooks-cheatsheet-386c05aeebed) will be a cheatsheet for Jupyter Notebook's formatting. 
