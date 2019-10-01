#!/usr/bin/env python
# coding: utf-8

# I created this workbook for educational purposes so please contribute to improve it further. With time, I will be adding more stuff on bias variance trade-off i.e. validation, cross validation, learning curves etc. so STAY TUNED!! For problem description, please visit [this](https://www.kaggle.com/c/titanic) webpage. 

# <h1 id="tocheading">Table of Contents</h1>
# <div id="toc"></div>

# In[ ]:


get_ipython().run_cell_magic(u'javascript', u'', u"$.getScript('https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js')")


# In[ ]:


import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic(u'matplotlib inline')
import re
from sklearn import *
import xgboost as xgb
from sklearn.svm import SVC


# ## Importing Data to Pandas Dictionary

# In[ ]:


train_df = pd.read_csv('../input/train.csv', index_col=False )
test_df = pd.read_csv('../input/test.csv', index_col=False )


# One more copy for testing purposes

# In[ ]:


train_df_new = pd.read_csv('../input/train.csv', index_col=False )


# ## Lets look at the data 

# In[ ]:


train_df.head()


# In[ ]:


train_df.describe()


# ## Lets visualize the relationship between the target variable and independent variables
# ### Categorical Variables:

# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(11.7, 10.27)
plt.figure(1)
plt.subplot(221)
plt.title('Ticket Class')
sb.set(style="darkgrid")
sb.countplot(x="Pclass", hue = "Survived", data = train_df,palette="Set3", edgecolor=sb.color_palette("husl", 8))
plt.subplot(222)
plt.title('Gender')
sb.set(style="darkgrid")
sb.countplot(x="Sex", hue = "Survived", data = train_df, palette="Set3", edgecolor=sb.color_palette("husl", 8))
plt.subplot(223)
plt.title('Port of Embarkation')
sb.set(style="darkgrid")
sb.countplot(x="Embarked", hue = "Survived", data = train_df,palette="Set3", edgecolor=sb.color_palette("husl", 8))
plt.subplot(224)
plt.title('Number of Siblings and Spouses')
sb.set(style="darkgrid")
sb.countplot(x="SibSp", hue = "Survived", data = train_df,palette="Set3", edgecolor=sb.color_palette("husl", 8) )


# We observed that ticket classes 1 & 2 had better survival than class 3. Also, females had better survival than males. People embarked from port C has better survival. 

# ### Continous Variables:

# In[ ]:


g = sb.PairGrid(train_df,
                 y_vars=["Age", "Fare"],
                 x_vars=["Survived"],
                 aspect=2, size=4)
g.map(sb.violinplot, palette="Set3");


# We observed that higher fare had better odds of being survived.

# ## Lets do some feature engineering

# We observed that most people had their title embedded in the name that could provide information on marital status, profession, etc. Titles like "Miss", "Mr", "Mrs" would be highly correlated with gender but we dont need to worry about it at this point. Lets mine this information.

# #### Getting title from name

# In[ ]:


train_df["title"] = [i[i.index(', ')+2:i.index('.')] for i in train_df["Name"]]


# In[ ]:


np.unique(train_df["title"])


# Mapping rare titles to 'rare'

# In[ ]:


rare_title = ['Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']


# In[ ]:


train_df["title"] = ["Rare" if i in rare_title else i for i in train_df["title"]]


# Lets visualize the new variable "Title"

# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(11.7, 10.27)
plt.title('Title')
sb.set(style="ticks")
sb.countplot(x="title", hue = "Survived", data = train_df, palette="Set3", edgecolor=sb.color_palette("husl", 8))


# ### Lets see what ticket number tells us

# Some ticket numbers have alpha charaters in the number. Assuming that those came from different vender, I removed these ticket numbers and kept the number only ticket numbers.

# In[ ]:


train_df["ticket_number"] = [i if re.search('[a-zA-Z]', i) == None else None for i in train_df["Ticket"]]


# Lets plot the new variable

# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(11.7, 10.27)
sb.stripplot(x="Survived", y="ticket_number", data=train_df, palette="Set3");


# Does not seem to be helpful so we will drop this variable from further analysis.

# I believe people with higher number of family members have better odds of Survival. Lets join Sibsip and Parch variable to create "Family" variable. 

# In[ ]:


train_df["family_members"] =  train_df["Parch"] + train_df["SibSp"]


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(11.7, 10.27)
plt.title('Number of Family Members')
sb.set(style="white")
sb.countplot(x="family_members", hue = "Survived", data = train_df, palette="Set3", edgecolor=sb.color_palette("husl", 8))


# ## Lets visualize the effect of covariates on probability of survival

# In[ ]:


def prob_surv(dataset, group_by):
    df = pd.crosstab(index = dataset[group_by], columns = dataset.Survived).reset_index()
    df['prob_surv'] = df[1] / (df[1] + df[0])
    return df[[group_by, 'prob_surv']]


# In[ ]:


train_df['age_cat'] = pd.cut(train_df['Age'], 30, labels = np.arange(1,31))
train_df['fare_cat'] = pd.cut(train_df['Fare'], 50, labels = np.arange(1,51))
sb.lmplot(data = prob_surv(train_df, 'age_cat'), x = 'age_cat', y = 'prob_surv', fit_reg = True, palette="Set3")
plt.title('Probability of being Survived with respect to Age')
plt.show()
sb.lmplot(data = prob_surv(train_df, 'fare_cat'), x = 'fare_cat', y = 'prob_surv', fit_reg = True, palette="Set3")
plt.title('Probability of being Survived with respect to Fare')
plt.show()


# We observed that survival decreases with increasing age and increases with incresing fare 

# ## Lets do some predictions now

# In[ ]:


train = train_df_new.drop(["Survived"], axis = 1)


# In[ ]:


all_df = pd.concat((train, test_df), axis=0, ignore_index=True)
all_df["family_members"] =  all_df["Parch"] + all_df["SibSp"]
all_df["title"] = [i[i.index(', ')+2:i.index('.')] for i in all_df["Name"]]
all_df["title"] = ["Rare" if i in rare_title else i for i in all_df["title"]]
all_df['mother'] = ['Mother' if all_df["Parch"][i]>0 and all_df["Sex"][i] == 'female' and all_df["title"][i] != 'Miss' else 'nonMother' for i in range(len(all_df))]
all_df['child'] = ['child' if all_df["Age"][i] < 18 else 'adult' for i in range(len(all_df))]


# ### Encode variables to numbers

# Lets take care of missing values first

# In[ ]:


from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


# In[ ]:


all_df = DataFrameImputer().fit_transform(all_df)


# ### Encoding variables

# In[ ]:


all_df['age_cat'] = (pd.cut(all_df['Age'], 30, labels = np.arange(1,31))).astype(int)
all_df['fare_cat'] = (pd.cut(all_df['Fare'], 50, labels = np.arange(1,51))).astype(int)
for i in all_df.columns:
    if all_df[i].dtype == 'object':
        if i in ['Sex', 'Embarked', 'title', 'mother', 'child']:
            le = preprocessing.LabelEncoder()
            all_df[i+'_new']= le.fit_transform(all_df[i].values)
            all_df.drop([i], axis = 1)
    



# In[ ]:


all_df = all_df.drop(["PassengerId", "Name", "Sex", "Ticket", "Cabin", 'child', "Embarked", "title", "mother"], axis = 1)


# Here is the final table ready for predictions

# In[ ]:


all_df.head()


# In[ ]:


train = all_df.iloc[:len(train)]
test = all_df.iloc[len(train):]
y = train_df['Survived'].values


# In[ ]:


pid = test_df["PassengerId"].values
np.unique(y)


# In[ ]:


all_df.shape


# In[ ]:


from sklearn.metrics import accuracy_score
fold = 10 
for i in range(fold):
    params = {
        'eta': 1,
        'max_depth': 15,
        'objective': 'binary:logistic',
        'seed': i+10,
        'lambda' : 3,
        'alpha' : 3
    }
    x1, x2, y1, y2 = model_selection.train_test_split(train, y, test_size=0.10, random_state=i)
    
    model = xgb.train(params, xgb.DMatrix(train, y), 100, verbose_eval=50)
    predictions_test = model.predict(xgb.DMatrix(x2))
    predictions = model.predict(xgb.DMatrix(test))
    survived = [int(round(value)) for value in predictions]
    survived_test = [int(round(value)) for value in predictions_test]
    accuracy = accuracy_score(y2, survived_test)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    #submission = pd.DataFrame(survived, columns=['Survived'])
    #submission["PassengerId"] = pid
    #submission.to_csv('submission_xgb_titanic'  + str(i) + '.csv', index=False)
    


# ### We got very good accuracy to start with.

# ### Lets look at the Feature Importance plot

# In[ ]:


plt.rcParams['figure.figsize'] = (8.0, 8.0)
xgb.plot_importance(booster=model); plt.show()



# ### Lets try SVC

# In[ ]:


x1, x2, y1, y2 = model_selection.train_test_split(train, y, test_size=0.10, random_state=i)


# In[ ]:


x1, x2, y1, y2 = model_selection.train_test_split(train, y, test_size=0.18, random_state=1)
model_svc = SVC(kernel='linear', probability=True)
model_svc.fit(x1, y1) 
predictions_test = model_svc.predict(x2)
accuracy = accuracy_score(y2, predictions_test)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# ### Lets try Random Forest

# In[ ]:


model_rf = ensemble.RandomForestClassifier(n_estimators=100)
model_rf.fit(x1, y1)
y_pred = model_rf.predict(x2)
model_rf.score(x2, y2)


# ## I hope you learnt something from this workbook. Ideas to improve are always welcome.

# ## Lastly, lets create submission file.

# In[ ]:


#submission_svc = pd.DataFrame(y_pred, columns=['Survived'])
#submission_svc["PassengerId"] = pid
#submission_svc.to_csv('submission_rf_titanic'  + str(i) + '.csv', index=False)


# In[ ]:


get_ipython().run_cell_magic(u'javascript', u'', u"$.getScript('https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js')")


# In[ ]:




