#!/usr/bin/env python
# coding: utf-8

# # Titanic data exploration

# I am going to do some data exploration of the titanic data. Data exploration is a very important step that should be taken carefully. This step leads to a better understanding of data.

# The necessary imports

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns


# Now let us get the training data and the test data

# In[ ]:


train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")


#     The first thing to do is to identify the ```predictors``` and the ```target```

# In[ ]:


train_data.columns


# In[ ]:


test_data.columns


# We can see that both ```train_data``` and ```test_data``` have the same columns except for ```Survived``` which is only in ```train_data```. This is not a surprise since we are supposed to use ```test_data``` to predict the values of ```Survived``` for each passenger. This is a big hint that we will be using a supervised learning to model our system.

# Next we are going to get the shapes of the training and the testing data

# In[ ]:


train_data.shape


# In[ ]:


test_data.shape


# From the shapes above, 
# * ```train_data``` has 891 rows and 12 columns. This implies that there are 891 training examples (891 passengers in this case). The 12 columns comprise of 11 training features and 1 target value (```Survived``` in this case)
# * ```test_data``` has 418 passengers and 11 features (Remember ```test_data``` does not have ```Survived```)

# Next we investigate the data types of the features and the target. We will only use the ```train_data``` as the data types of the features of ```test_data``` are the same as those of ```test_data```

# In[ ]:


train_data.info()


# From the  data types above:
# * ```Pclass```, ```Age```, ```SibSp```, ```Parch```, ```Fare```, ```Survived``` and ```PassengerId``` are either ```int64``` or ```float64``` types. This shows that they are numerical types.
# * ```Name```, ```Sex```, ```Ticket```, ```Cabin``` and ```Embarked``` are catagorical types. Depending on the learnig algorithm used, these features may need to be converted to numerical types

# I am now going to explore the numerical and categorical features differently. I will do *univariate* and *bivariate* analysis

# ## Univariate analysis

# Univariate analysis involves exploring features one by one

# I will start by investigating numerical features first then later handle categorical features

# ### Numerical data univariate analysis

# Let's get the description of the data

# In[ ]:


train_data.describe()


# Using ```describe``` function of the dataframe, we get the following:
# * count - this is the total number of examples (passengers) in the ```train_data```. There are 891 examples, therefore all the features and the target are given in 891 points. For the ```train_data```, the count would be 418.
# * The ```mean```, ```min``` and ```max``` are self explanatory
# * ```std``` is the standard deviation, which shows how data is dipersed
# * ```25%```, ```50%``` and ```75%``` are 25th percentile, 50th percentile and 75th percentile respectively. 50th percentile is the median.
# 
# I am going to give some light about a few properties below
# 1. For ```Pclass```, mean (*2.308642*) is lower than the median (*3.0000*). This shows that the classes of the passenger was skewed more towards 3rd class, as most people were on the 3rd class.
# 2.  For ```Fare```, mean (*32.204208*) is higher than the median (*14.454200*). This shows that the fares were skewed more towards lower fares, as most people were on the 3rd class, agreeing with the observation above.
# 3. For ```Age```, the mean (*29.699118*) is almost equal to the median (*28.000000*). This shows that the data is almost evely distributed on the right and left side of the median
# 4. For ```SibSp```, the mean is (*0.523008*) is higher than the median(*0.0000*). The skew is to the left as many passengers were travelling without a spouse or sibling. Almost the same observation can be made for ```Parch``` as many passengers did not have a parent or child onboard.
# 5. For ```Survived```, mean (*0.383838*) is higher than the median (*0.0000*). This is because many passengers died, thus skew was to the left.

# #### Visual univariate analysis for numerical values

# ##### Histograms

# Histograms show how numerical values are distributed in consecutive but non-overlapping buckets.

# I will start with the Histogram for ```Pclass```. This histogram should lead to same conclusions that I made above on the distribution of ```Pclass```.

# In[ ]:


sns.distplot(train_data['Pclass'], bins=3, kde=False)


# It can clearly be seen that most of the passengers were in the third class, therefore leading to a skew to the right. This agrees with the conclution we made earlier!

# Next we plot the histogram for ```Fare```

# In[ ]:


sns.distplot(train_data['Fare'], bins=50, kde=False)


# From the plot, the skew is to the left as most people paid small fares. This again agrees with the conclusion earlier.

# Now is time to have a look at the plot for ```Age```

# In[ ]:


sns.distplot(train_data['Age'].dropna(), kde=False)


# Looking at the plot above, ages are almost evenly distributed as concluded earlier.

# How about ```SibSp``` and ```Parch```?

# In[ ]:


sns.distplot(train_data['SibSp'], bins=6, kde=False)


# In[ ]:


sns.distplot(train_data['Parch'], bins=6, kde=False)


# The 2 lots above agree that many people travelled alone

# Last we plot for ```Survived```

# In[ ]:


sns.distplot(train_data['Survived'], bins=2, kde=False)


# Clearly many people did not survive.

# ##### Boxplots

#  A boxplot is used to graphically represent the quartiles of the data. It has a rectangle that shows the second and third quartile with a line passing through the box showing the media.  Lines on either sides of the rectangle indicate the lower and upper quartiles. Outliers are shown as dots outside the quartiles.

# Lets look at the plot for ```Pclass```

# In[ ]:


sns.boxplot(train_data['Pclass'])


# The boxplot shows that second to fourth quartiles are between 2 and 3. Only a quarter of the passengers were in the first class and the rest were in second or third class

# Now is time to look at ```Fare```

# In[ ]:


sns.boxplot(train_data['Fare'])


# Generally the fare was less than 100, that is why we see the 4 quartiles below 100. The higher fares are generally viewed as outliers. Some actually paid over 500!

# It is time to look at ```Age```

# In[ ]:


sns.boxplot(train_data['Age'])


# Remember the conclusions above that age is almost evely distributed? The boxplot suggests that too, except for a outliers above the fourth quartile. The outliers represent a few older passengers

# Box plot for ```Parch``` is shown below

# In[ ]:


sns.boxplot(train_data['Parch'])


# Again we can see that anayone who had a child or a parent onboard is treated as an outlier as all the quartiles are actually at 0!

# How about ```SibSp```?

# In[ ]:


sns.boxplot(train_data['SibSp'])


# The first to third quartiles are between 0 and 1, indicating that about 3/4 of the passengers were either alone or with a spouse or a sibling. It was however found out earler that most of the people did have a spouse or sibling as the median for this feature was 0. Anyone with more than 2 spouses or child is treated as an outlier. The media is at 0 here.

# Lastly I do the plot for our target variable, ```Survived```

# In[ ]:


sns.boxplot(train_data['Survived'])


# No surprise al the quartiles are between 0 and 1 with no outliers. Thought not clearly seen, the median is at 0

# ### Categorical data univariate analysis

# I have taken some time exploring the numerical variables, next I am going to shift to categorical variables. 

# Let me remind us the categorical variables by checking their first 5 values.

# In[ ]:


train_data.select_dtypes(include=['object']).head()


# Now that we remember the categorical variables, I am going to do univariate analysis of them. Let go!

# ##### Countplot

# I will start by doing a *countplot* for these features which is like a histogram for categorical variables

# I will begin with ```Sex```

# In[ ]:


sns.countplot(x="Sex", data=train_data)


# Male were almost twice the number of female.

# Lets check the distribution of ```Emarked```

# In[ ]:


sns.countplot(x="Embarked", data=train_data)


# We see that many people embarked at Southampton. then Cherbourg and very few embarked at Queenstown.

# I am not going to do the plots for other categorical variables as they are very complex since they involve very many unique variables

# Next I shift to *bivariate analysis*

# ## Bivariate analysis

# Finds the relationship between 2 variables

# As I did above, I am going to start with continuous values. One of the most common plots is scatter plot, which shows the correlation between continous values. We can use these plot to see if there is a correlation between the 2 variables. Unfortunatly many of the variables are not really continous,  only ```Age``` and ```Fare``` seem to be really continous. I will however attempt to deduce correlation from some of the discrete variables.

# ### Bivariate analysis for continous values

# ##### Scatterplots

# Scatter plots are used to show how variables relate with each other

# In[ ]:


train_data.plot.scatter(x="Fare", y="Pclass")


# From the plot, is is obvious that the upper class passengers paid more fare than lower class ones. This an expected behaviour

# In[ ]:


train_data.plot.scatter(x="Fare", y="Age")


# Is there a strong relationship between ```Fare``` and ```Age```? Unfortunately, I really do not see a strong relation clearly.

# In[ ]:


train_data.plot.scatter(x="Fare", y="Parch")


# Generally many people did not travel with parents or children, although there seems to be a 'weak' pattern of low fares with more children than higher ones. We however saw earlier that those with more children or parents were outliers. A weak relation like this can also be seen below but for ```Parch``` and ```Age```

# In[ ]:


train_data.plot.scatter(x="Age", y="Parch")


# In[ ]:


train_data.plot.scatter(x="Fare", y="SibSp")


# Again people with more than 1 sibling or spouse were seen as outliers. However for those outliers, lower fares were associated with more children or spouses

# I am not going to do further scatter plots. I will later do a correlation between all the variables a little later.

# ### Bivariate analysis for discrete and categorical features

# For categorical and discrete values we can do count plot of 2 variables

# In[ ]:


sns.countplot(x=train_data["Pclass"], hue=train_data["Survived"])


# Genarally, the upper the class the passenger was in, the higher the likelihood of survival. For this dataset, more people survived than those who died in the first class. For second and third classes, more died than those who survived but the ratio of survivors to total passengers in second class was higher than that in third class. However, the number of survivors in the 3 classes was roughly the same. What could this mean?

# In[ ]:


sns.countplot(x=train_data["Parch"], hue=train_data["Survived"])


# About 2/3 of passengers who did not have a parent or a child onboard lost their lives. The ratios are very comparable for people who had a parent or child onboard, in fact for those that had the numbers seem to be the same. Did having someone someone you knew onboard increased your chance of survival?

# In[ ]:


sns.countplot(x=train_data["SibSp"], hue=train_data["Survived"])


# The observation here is almost the same as that for ```Parch```

# In[ ]:


sns.countplot(x=train_data["Sex"], hue=train_data["Survived"])


# This is interesting! Females who survived were much more than those who survived. For male, the inverse is true. Were ladies better at attracting the resque team or the resque team was intentionally looking for ladies first? I would what would happen in case there was a chance to only save one person but there were a female and a male who needed help.

# In[ ]:


sns.countplot(x=train_data["Embarked"], hue=train_data["Survived"])


# Only passengers that embarked in Cherbourg had more survivors than those who perished

# I will stop here for *countplots*

# **Are there any missings values**

# Next I am going to check if some values were missing. Some values may be missing because they were not available during the recording or they were available but were ignored either intensionally or unintentionally. Values may have been recorded but some other reasons may lead to the loss of the data. We are going to count the number of points that each feature did not have a value

# In[ ]:


train_data.isnull().sum()


# The missing data points are shown above. Most of the values for the cabin are not available. This feature should be safely removed from the training set before analysis. For age and embarked, the missing values can easily generated from the available values. The rest of the features are okay.

# This is just my first step in exploring this dataset. I will be improving this kernel with time. Your feedback and suggestions are very welcome
