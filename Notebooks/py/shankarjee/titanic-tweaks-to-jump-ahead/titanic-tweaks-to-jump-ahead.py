#!/usr/bin/env python
# coding: utf-8

# I am new to analytics and I think I pretty much struggle with the same questions which a newbie would have. I am not saying I have found the answers. I just want to share what a newbie might like to see

# In[ ]:


# Import the friendly libraries. 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns


# In[ ]:


train = pd.read_csv('../input/train.csv')
train.info()


# Making sense of data is probably the most important step. So this is my though process
# 1. **PassengerId** is some numbering system, hence I think it is not important at all. So we can ignore it
# 1. **Survived** is the output we want to predict
# 1. **Pclass** I think is like level where your cabin might be located. So higher level puts one at higher risk.
# 1. **Name**. This is important as it can help us decode the sex and to some extent age of the person
# 1. **SibSp** and **Parch** tells us information if a person has any dependents. So is the person alone or not. 
# 1. **Ticket** is just a number, so not importnant
# 1. **Fare** is a probably a good indicator of what class the person travels. Higher clas probably has more life boats
# 1. **Cabin** might be useful. However we have lot of null values so I decided to skip it
# 1. **Embarked**. So maybe the passengers get seated based on where they embark. 
# 
# We can check this based on correlatation matrix
# 

# In[ ]:


cm = train.corr()
print(cm['Survived'].sort_values(ascending= False))


# As expected the PassenderId has no influence on the survival rate. Now time to clean up the data. I wrote a subroutine for this which does the following
# 1. Drop unwanted columns
# 1. Map Sex to 0 and 1
# 1. Fill the NaN values in fare with the median values. We can also use mean, The effect was minimal
# 1. Map embarked location to integer again. So to get the unique values for a column say Sex one can do *train.Sex.unique()*. 
# 1. To fillin the Nan's in age first I find the title of every passenger. I store this in titles. The option *value-counts* gives us the unique values and how many times it appears.
#     1. I mapped titles like 'Dr', 'Sir', 'Don', 'Capt', 'Major', 'Rev', 'Col', 'Jonkheer' to Mr. Note we need to have regex = True turned on to do this
#     1. Similarly 'Ms', 'Mlle' to Ms and
#     1. 'Lady', 'the Countess', 'Mme' to Mrs
# 1. Now with this computed the median age of Mr, Mrs, Ms and Master. Use this to fill Nan values
# 1. Instead of using SibSp and Parch I created a new column called Relatives which is sum of SibSp and Parch. A person is also if relatives is 0. After creating this column I deleted SibSp and Parch
# 1. I split the passengerId and rest of the data and return a tuple of both
# 

# In[ ]:


def readAndCleanUpData(fileName):    
    df = pd.read_csv(fileName)

    # Drop ticket and cabin
    df.drop(['Ticket', 'Cabin'], axis = 1, inplace = True)

    # Replace male with 0, female with 1
    df.Sex = df.Sex.map( {'male': 0, 'female': 1} )
    
    # Fill fare with median value
    df['Fare'].fillna( df.Fare.median(), inplace = True)

    # In Embarked we have 3 null values. We will replace this with median values
    # First convert to 0 1 and 2
    df.replace( {'S': 0, 'C':1, 'Q':2}, inplace = True)
    df['Embarked'].fillna( df['Embarked'].median(), inplace = True)    

    # Get title of all passengers
    titles = df.Name.str.split(',').str.get(1).str.split('\.').str.get(0)
    titles.value_counts()
    
    # We have Dr, Sir, Don, Capt, major, Rev. Replace, Jonkheer with Mr    
    df.Name.replace(['Dr', 'Sir', 'Don', 'Capt', 'Major', 'Rev', 'Col', 'Jonkheer'], 'Mr', regex = True, inplace = True)        
    # Replace Ms, Mlle with Miss
    df.Name.replace(['Ms', 'Mlle'], 'Miss', regex = True, inplace = True)    

    # Replace Lady, Countess, Mme with Mrs
    df.Name.replace(['Lady', 'the Countess', 'Mme'], 'Mrs', regex = True, inplace = True)        

    # Now get mediam mr, mrs, master ages and replace na with mean
    idx = df['Name'].str.contains('Mr\.')
    median = df.Age[idx].mean()
    df.loc[ idx, 'Age'] = df.loc[ idx, 'Age'].fillna(median)

    idx = df['Name'].str.contains('Mrs\.')
    median = df.Age[idx].median()
    df.loc[ idx, 'Age'] = df.loc[ idx, 'Age'].fillna(median)

    idx = df['Name'].str.contains('Miss\.')
    median = df.Age[idx].median()
    df.loc[ idx, 'Age'] = df.loc[ idx, 'Age'].fillna(median)

    idx = df['Name'].str.contains('Master\.')
    median = df.Age[idx].median()
    df.loc[ idx, 'Age'] = df.loc[ idx, 'Age'].fillna(median)
    
    df.Age = df.Age.astype(int)
    
    # Create a new column on relatives
    df['Relatives'] = df['SibSp'] + df['Parch']
    df['isAlone'] = 0
    df.loc[ df.Relatives == 0, 'isAlone'] = 1

    # We can delete the name column now    
    pId= df.PassengerId
    df = df.drop(['Name', 'SibSp', 'Parch', 'Relatives', 'PassengerId'], axis = 1)
    
    # Normalize Fare    
    return df, pId


# Now get cleaned up test and train data

# In[ ]:


# Read in training data
df, pId = readAndCleanUpData('../input/train.csv')
# Read in test data
t_df, pId = readAndCleanUpData('../input/test.csv')
nTrain = df.shape[0] # This is number of training data we have

# Get surival data from train set
y     = df.Survived


# We need to scale the data. I used standardscaler. To scale the date I combined test and train data. This is what *nTrain* is important. Once we get the results we need to split it again for further analysis

# In[ ]:


allData = pd.concat( [df.drop(['Survived'], axis = 1), t_df] )
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
colNames = list( allData.columns.values )
allData_Scaled = pd.DataFrame( scaler.fit_transform(allData), columns = colNames )
# Split back now
X = allData_Scaled.iloc[:nTrain, :]
t_df_c = allData_Scaled.iloc[nTrain:, :]


# We will visualize the correlation matrix using seaborn. To get the numbers to display on heatmap we need to turn on annot, To scale the font size we need to use sns.set(). Found those pretty useful. Also to change the plot size we need to use matplotlib handles. We create a subplot where we input the figsize. Use the axis output from this and pass it on to heatmap. 

# In[ ]:


# Correlation matrix
cm = X.corr()
get_ipython().magic(u'matplotlib inline')
import matplotlib.pylab as plt
sns.set(font_scale=1.2)
fig, ax = plt.subplots(figsize = (10,10))
sns.heatmap(ax = ax, data = cm, vmax = 0.8, square = True, fmt = '.2g', annot = True)
plt.show()


# Now comes the part of fitting it. Just tweaking the parameters inside the classifier makes a **big difference**. I don't claim to have the mastery of all options. But this helped me to jump ahead

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
params = {'bootstrap': False, 'min_samples_leaf': 5, 'n_estimators': 80, 'min_samples_split': 10, 
              'max_features': 'auto', 'max_depth': 20}
model = RandomForestClassifier(**params)
model.fit(X, y)


# In[ ]:


# A subroutine to compute the accuracy. Maybe scikit has it
from sklearn.model_selection import cross_val_score
def compute_accuracy(model, X, y):
    return np.mean( cross_val_score(model, X, y, cv = 5, scoring='accuracy') )  
print( "Accuracy of the model ", compute_accuracy(model, X, y) )


# Now comes the part of predicting and saving data

# In[ ]:


# Predictions
y_pred = model.predict( t_df_c )

# Create predictions
predictions =  pd.DataFrame( {'PassengerId' : pId,
                             'Survived'    : y_pred} )

# Save the output
predictions.to_csv("my_predictions.csv", index = False)


# I am hoping this helps someone also me. I wish to learn from your comments

# In[ ]:




