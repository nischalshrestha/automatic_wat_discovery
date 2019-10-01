#!/usr/bin/env python
# coding: utf-8

# Importing required libraries

# In[283]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


# Loading the dataset

# In[129]:


dataset = pd.read_csv('../input/train.csv')
backup_set = dataset[['PassengerId', 'Survived']]


# # Exploration Begins

# In[130]:


dataset.info()


# We got some missing values here. **Age, Embarked, Cabin** columns contain missing values. **Cabin** column have a lot of missing values, So we will drop it. In **Embarked**, only two values are missing.

# In[131]:


dataset = dataset.drop('Cabin', axis=1)
dataset.head()


# **Data Dictionary**
# * **survival**: Survival - *0 = No, 1 = Yes* 
# * **pclass**: Ticket class - *1 = 1st, 2 = 2nd, 3 = 3rd*
# * **sex**: Sex	
# * **Age**: Age in years	
# * **sibsp**: # of siblings / spouses aboard the Titanic	
# * **parch**: # of parents / children aboard the Titanic	
# * **ticket**: Ticket number	
# * **fare**: Passenger fare	
# * **cabin**: Cabin number	
# * **embarked**: Port of Embarkation - *C = Cherbourg, Q = Queenstown, S = Southampton*
# 
# **Variable Notes**
# * **pclass**: A proxy for socio-economic status (SES)
#     * 1st = Upper
#     * 2nd = Middle
#     * 3rd = Lower
# 
# * **age**: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# * **sibsp**: The dataset defines family relations in this way...
#     * Sibling = brother, sister, stepbrother, stepsister
#     * Spouse = husband, wife (mistresses and fiancés were ignored)
# 
# * **parch**: The dataset defines family relations in this way...
#     * Parent = mother, father
#     * Child = daughter, son, stepdaughter, stepson
#     * Some children travelled only with a nanny, therefore parch=0 for them.

# In[132]:


tot_surv_died = dataset['Survived'].value_counts().values
plt.pie(labels=['Died', 'Survived'], x=tot_surv_died/891, autopct='%1.1f%%', colors=['red','green'])
plt.axis('equal')
plt.show()


# In[133]:


print('Died = 0, Survived = 1')
print(dataset['Survived'].value_counts())


# Majority of the passengers have died. Just a little more than one third of passengers survived.

# In[134]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey = True)
tot_pass_by_sex = dataset['Sex'].value_counts().values
ax1.bar(['Male', 'Female'], tot_pass_by_sex)
ax1.set_title('Number of Passengers')
surv_by_sex = dataset[dataset['Survived'] == 1]['Sex'].value_counts().values
ax2.bar(['Male', 'Female'], surv_by_sex[::-1])
ax2.set_title('Survived')
died_by_sex = dataset[dataset['Survived'] == 0]['Sex'].value_counts().values
ax3.bar(['Male', 'Female'], died_by_sex)
ax3.set_title('Died')
plt.show()


# In[135]:


print("Number of Passengers")
print(dataset['Sex'].value_counts())
print("\nSurvived")
print(dataset[dataset['Survived'] == 1]['Sex'].value_counts())
print("\nDied")
print(dataset[dataset['Survived'] == 0]['Sex'].value_counts())


# A higher proportion of females survived compared to males. Even though female count is only around half of male count, female survival count is two times more than male survival count. This clearly shows that this is not a result of mere chance. Females were given more priority than male. Now let's check which age categories have better survival count. But for that we need to fill those missing values in **Age** column. Well we can either fill it with a mean value, median, value, or a most frequent value. We can also remove rows which have missing values but since this is a small dataset we won't be removing anything. Or we can do something better. Why don't we predict **Age** column? That seems to be pretty good. Let's give it a go.

# In[136]:


dataset[dataset['Age'].notnull()].info()


#  **Embarked** have only two missing values so we will fill it with most frequent values as it doesn't make sense to predict just two values.

# In[137]:


dataset.loc[dataset['Embarked'].isnull(), 'Embarked'] = dataset.loc[dataset['Embarked'].notnull(), 'Embarked'].mode().values
dataset[dataset['Age'].notnull()].info()


# In[138]:


dataset[dataset['Age'].notnull()].head()


# Okay so that's done. We know that **PassengerId** got nothing to do with this data and our test dataset doesnot contain **Survived** column so we won't use both of these columns for prediction. Then, we need to create dummy variables for **Pclass**, **Sex** and **Embarked** columns.

# The **Ticket** column isn't giving us  any valuable information so let's drop it.

# In[139]:


dataset = dataset.drop('Ticket', axis=1)


# The **Name** column looks like it isn't valuable but the title in the name could be valuable to us. So let's extract title from the **Name** column.

# In[140]:


dataset['Title'] = dataset['Name'].str.extract('(\w+(?=\.))', expand=False)
dataset = dataset.drop('Name', axis=1)
dataset.head()


# Let's check those titles.

# In[141]:


dataset['Title'].value_counts()


# Mlle is an abbreviation of Mademoiselle traditionally given to an unmarried woman. Mme is an abbreviation of Madame which is given to women where their marital status is unknown. Ms is to refer to a women irrespective of their marital status. So we will change them to Miss. Dr, Rev, Col, Major, Sir, Capt are occupation related titles so we will change it to Occupation.  Don, Countess, Lady, Jonkheer are noble or honorific titles so we will change it to Noble

# In[142]:


backup_set['Title'] = dataset['Title']
dataset['Title'] = dataset['Title'].str.replace('(Mlle|Mme|Ms)', 'Miss')
dataset['Title'] = dataset['Title'].str.replace('(Dr|Rev|Col|Major|Sir|Capt)', 'Occupation')
dataset['Title'] = dataset['Title'].str.replace('(Don|Countess|Lady|Jonkheer)', 'Noble')
dataset['Title'].value_counts()


# We can create new columns using **SibSp** and **Parch** and that is:
# * **IsAlone** - Whether a passenger boarded alone or is with family
# * **FamilyCount** - Number of family members a passenger have on-board.
# 
# We can also create a new column using **Title** and **Parch** and that is:
# * **IsMother** - Female who is with greater than 0 value in **Parch** and have a **Title** of Mrs.

# In[143]:


dataset['FamilyCount'] = dataset['SibSp'] + dataset['Parch']
dataset['IsAlone'] = 0
dataset.loc[dataset['FamilyCount'] == 0, 'IsAlone'] = 1
dataset['IsMother'] = 0
dataset.loc[(dataset['Sex'] == 'female') & (dataset['Parch'] > 0) & (dataset['Title'] == 'Mrs'), 'IsMother'] = 1
dataset.head(10)


# Let's convert **Sex** column to numeric. 0 for male and 1 for female.

# In[144]:


dataset.loc[dataset['Sex'] == 'male', 'Sex'] = 0
dataset.loc[dataset['Sex'] == 'female', 'Sex'] = 1
dataset.head()


# Let's create a **FareBand** column instead of our **Fare** column.

# In[145]:


dataset['Fare'].describe()


# In[146]:


pd.cut(dataset['Fare'], bins=[0, 8, 15, 32, 100, 600]).value_counts()


# In[147]:


dataset['FareBand'] = pd.cut(dataset['Fare'], bins=[0, 8, 15, 32, 100, 600], labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
dataset = dataset.drop('Fare', axis=1)


# Let's check out the survival rate of passengers within this fare band.

# In[148]:


fare_band_surv_rate = dataset[dataset['Survived'] == 1]['FareBand'].value_counts().values / dataset['FareBand'].value_counts().values * 100
plt.bar(['Very Low', 'Low', 'Medium', 'High', 'Very High'], fare_band_surv_rate)
plt.title('Survival Rate (Ticket Fare)')
plt.show()


# In[149]:


dataset[dataset['Survived'] == 1]['FareBand'].value_counts().values / dataset['FareBand'].value_counts().values * 100


# Hm... Value for money I guess. 73.58% of passengers who bought expensive tickets survived. Now, lets do a few more analsis before we move onto create dummy variables for our **Embarked** and **Title** columns

# In[150]:


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8,8))
ax1.hist([dataset.loc[:,'FamilyCount'], dataset.loc[dataset['Survived'] == 1, 'FamilyCount']],bins=np.arange(12)-0.5, color=['blue', 'green'], histtype='bar', label=['Family Members On-Board', 'Family Members Survived'])
ax1.set_xticks([0,1,2,3,4,5,6,7,10])
ax1.legend()
ax1.set_title('Survival Count (Family Count)')
ax2.hist([dataset.loc[:,'FamilyCount'], dataset.loc[dataset['Survived'] == 0, 'FamilyCount']],bins=np.arange(12)-0.5, color=['blue', 'red'], histtype='bar', label=['Family Members On-Board', 'Family Members Died'])
ax2.set_xticks([0,1,2,3,4,5,6,7,10])
ax2.legend()
ax2.set_title('Death Count (Family Count)')
plt.show()


# In[151]:


dataset.loc[:,'FamilyCount'].value_counts()


# In[152]:


dataset.loc[dataset['Survived'] == 1, 'FamilyCount'].value_counts()


# In[153]:


dataset.loc[dataset['Survived'] == 0, 'FamilyCount'].value_counts()


# Being Single hurts! But having a lot of family members on-board hurts too. Passengers who have 2-3 Family members on board have better survival rate.

# In[154]:


mother_surv_died =  dataset.loc[dataset['IsMother'] == 1, 'Survived'].value_counts().values
plt.pie(labels=['Died', 'Survived'], x=mother_surv_died[::-1]/56, autopct='%1.1f%%', colors=['red','green'])
plt.axis('equal')
plt.show()


# In[155]:


dataset.loc[dataset['IsMother'] == 1, 'Survived'].value_counts()


# As expected, Majority of mother's survived.

# In[156]:


plt.figure(figsize=(6,12))
plt.bar(['Master', 'Miss', 'Mr', 'Mrs', 'Noble', 'Occupation'], dataset.loc[:, 'Title'].value_counts().sort_index().values, color='blue')
plt.bar(['Master', 'Miss', 'Mr', 'Mrs', 'Noble', 'Occupation'], dataset.loc[dataset['Survived'] == 1, 'Title'].value_counts().sort_index().values, color='green')
plt.legend()
plt.title('Survival Count (Title)')
plt.show()


# In[157]:


dataset.loc[:, 'Title'].value_counts().sort_index()


# In[158]:


dataset.loc[dataset['Survived'] == 1, 'Title'].value_counts().sort_index()


# Again, Survival rate of females are really good. Out of 4 nobles 2 of them managed to survive. I bet those two are Lady and Countess. Let's take a look.

# In[159]:


backup_set[(backup_set['Title'] == 'Lady') | (backup_set['Title'] == 'Countess')]


# Voila! Females survived again. Now it's **Pclass**'s turn.

# In[160]:


plt.bar(['(1) Upper', '(2) Middle', '(3) Lower'], dataset.loc[:, 'Pclass'].value_counts().sort_index().values, color='blue')
plt.bar(['(1) Upper', '(2) Middle', '(3) Lower'], dataset.loc[dataset['Survived'] == 1, 'Pclass'].value_counts().sort_index().values, color='green')
plt.legend()
plt.title('Survival Count (Pclass)')
plt.show()


# In[161]:


dataset.loc[:, 'Pclass'].value_counts().sort_index()


# In[162]:


dataset.loc[dataset['Survived'] == 1, 'Pclass'].value_counts().sort_index()


# The class distinction is clear here. More than half of Upper class (1) passengers survived while 75.76% of Lower class (3) passengers died. Let's check **Embarked** column.

# In[163]:


plt.bar(['C', 'Q', 'S'], dataset.loc[:, 'Embarked'].value_counts().sort_index().values, color='blue')
plt.bar(['C', 'Q', 'S'], dataset.loc[dataset['Survived'] == 1, 'Embarked'].value_counts().sort_index().values, color='green')
plt.legend()
plt.title('Survival Count (Embarked)')
plt.show()


# In[164]:


dataset.loc[:, 'Embarked'].value_counts().sort_index()


# In[165]:


dataset.loc[dataset['Survived'] == 1, 'Embarked'].value_counts().sort_index()


# More than half of passengers from Cherbourg survived.  More than half of Queenstown passengers died and 66% of Southampton passengers died. Most probably the embarked port does not have any relation to Survival. Now let's create dummy variable for **Embarked** and **Title** columns.

# In[166]:


dataset = pd.concat([dataset, pd.get_dummies(dataset['Embarked'], drop_first=True), pd.get_dummies(dataset['Title'], drop_first=True)], axis=1).drop(['Embarked', 'Title'], axis=1)
dataset.head()


# Now let's convert **FareBand** column to numeric value.

# In[167]:


dataset['FareBand'] = dataset['FareBand'].cat.codes
dataset.head()


# Time to predict **Age** Column. Let's create a dataset from rows where **Age** is not null.

# In[168]:


age_set = dataset.loc[dataset['Age'].notnull(), ['PassengerId', 'Survived', 'Pclass', 'Sex', 'SibSp', 'Parch',
       'FamilyCount', 'IsAlone', 'IsMother', 'FareBand', 'Q', 'S', 'Miss',
       'Mr', 'Mrs', 'Noble', 'Occupation', 'Age']] 
age_set.info()


# In[169]:


X = age_set.loc[:, 'Pclass':'Occupation']
y = age_set['Age']

X_sm = sm.add_constant(X)
ols = sm.OLS(endog=y, exog=X_sm)
ols.fit().summary()


# We are going to perform Backward Elimination where we will remove columns with higher p-value than a set significance level (0.05). After removing we will refit the model again and again to our new dataset until no column is found to have a p-value higher than our significance level. Provided that the R-Squared measure doesn't get reduced during the process.

# In[170]:


def backwardelimination(y, X, SL=0.05, add_const=True):
    if(add_const == True):
        X = sm.add_constant(X)
    num_vars = X.shape[1]
    temp = pd.DataFrame(np.zeros(X.shape).astype('int'), columns=X.columns.values)
    temp = temp.set_index(X.index.values)
    for i in range(num_vars):
        ols_regressor = sm.OLS(endog=y, exog=X).fit()
        max_var = max(ols_regressor.pvalues)
        adj_rsquared_before = ols_regressor.rsquared_adj
        if(max_var > SL):
            for j in range(num_vars - i):
                if(ols_regressor.pvalues[j] == max_var):
                    temp.iloc[:,j] = X.iloc[:,j]
                    X = X.drop(X.columns[j], axis=1)
                    temp_regressor = sm.OLS(endog=y, exog=X).fit()
                    adj_rsquared_after = temp_regressor.rsquared_adj
#                     if(adj_rsquared_after <= adj_rsquared_before):
#                         X_rollback = pd.concat([X, temp.iloc[:,j]], axis=1)
#                         print(temp.iloc[:,j:j+1].head())
#                         print(ols_regressor.summary())
#                         return X_rollback
        else:
            print(ols_regressor.summary())
            return X
        
X_opt = backwardelimination(y, X)
X_opt.head()


# Let's use our new optimal dataset to predict age.

# In[171]:


X_train, X_test, y_train, y_test = train_test_split(X_opt.iloc[:,1:], y, test_size=0.2, random_state=1)


linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
y_pred = linear_regressor.predict(X_test)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
print(rmse)


# An RMSE of 10. Well this is not good but still its way better than filling it with mean, median, or mode. Let's train the model again with full set and use it predict null values in **Age** column.

# In[172]:


lin_reg_age = LinearRegression()
lin_reg_age.fit(X_opt.iloc[:,1:], y)

X_test_cols = ['Pclass', 'IsAlone', 'IsMother', 'Q', 'Miss', 'Mr', 'Mrs', 'Noble', 'Occupation']
dataset.loc[dataset['Age'].isnull(), 'Age'] = lin_reg_age.predict(dataset.loc[dataset['Age'].isnull(), X_test_cols])
dataset.info()


# Cool no missing values. But let's check for discrepancies as our predictions RMSE was 10.

# In[173]:


dataset['Age'].describe()


# In[174]:


dataset[dataset['IsMother'] == 1]['Age'].describe()


# In[175]:


dataset[dataset['Mrs'] == 1]['Age'].describe()


# Bingo! Youngest married woman is 14 years old? This could be an error in our prediction. Let's check **age_set** which contains non null **Age** values to verify it.

# In[176]:


age_set[age_set['Mrs'] == 1]['Age'].describe()


# Okay so we got a 14 year old married women here. And this is not an error in our prediction. Now let's do some analysis of our **Age** column.

# In[177]:


dataset['AgeBand'] = pd.cut(dataset['Age'], bins=[0, 1, 12, 18, 21, 29, 36, 60, 100])
dataset = dataset.drop('Age', axis=1)
age_band_surv_rate = dataset[dataset['Survived'] == 1]['AgeBand'].value_counts().sort_index().values / dataset['AgeBand'].value_counts().sort_index().values * 100
plt.figure(figsize=(8,4))
plt.bar(['(0, 1]', '(1, 12]', '(12, 18]', '(18, 21]', '(21, 29]', '(29, 36]', '(36, 60]', '(60, 100]'], age_band_surv_rate, color='green')
plt.title('Survival Rate (Age)')
plt.show()
dataset['AgeBand'] = dataset['AgeBand'].cat.codes


# Children upto age 12 have better survival rate. Most of the infants up to 1 year old survived.

# # Prediction
# Now it's time to do actual prediction. Let's take a look at our dataset.

# In[178]:


dataset.head()


# In[179]:


X = dataset.iloc[:, 2:]
y = dataset.iloc[:, 1]


# Let's try out various models with hyperparameter optimizations to find a model with best accuracy.
# 
# **Logistic Regression**

# In[180]:


log_reg = LogisticRegression(random_state=1)
selector = RFECV(log_reg, cv=5)
selector.fit(X, y)
selector.support_


# Seems like RFECV chose to keep all our features.  Let's do a grid search with all our features.

# In[190]:


parameters = {
    'solver': ['newton-cg', 'lbfgs', 'liblinear'],
}
gs_cv_lr = GridSearchCV(log_reg, param_grid=parameters, scoring='accuracy', cv=5, n_jobs=4)
gs_cv_lr.fit(X, y)
gs_cv_lr.best_score_


# **GaussianNB**

# In[222]:


g_nb_clas = GaussianNB()
np.mean(cross_val_score(g_nb_clas, X, y, scoring='accuracy', cv=5))


# **Stochastic Gradient Descent**

# In[252]:


sgd_clas = SGDClassifier(random_state=1, max_iter=1000, tol=None)
selector = RFECV(sgd_clas, cv=5)
selector.fit(X, y)
sgd_cols = X.columns[selector.support_].values


# In[260]:


parameters = {
    'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
    'penalty': ['l2', 'l1', 'elasticnet']
}
gs_cv_sgd = GridSearchCV(sgd_clas, param_grid=parameters, scoring='accuracy', cv=5, n_jobs=4)
gs_cv_sgd.fit(X[sgd_cols], y)
gs_cv_sgd.best_score_


# **K-Nearest Neighbours**

# In[242]:


kn_clas = KNeighborsClassifier()
parameters = {
    'n_neighbors': range(1,21),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p': [1, 2]
}
gs_cv_kn = GridSearchCV(kn_clas, param_grid=parameters, scoring='accuracy', cv=5, n_jobs=4)
gs_cv_kn.fit(X, y)
gs_cv_kn.best_score_


# **Support Vector Machine**

# In[255]:


sv_clas = SVC(random_state=1)
parameters = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'probability': [True, False],
    'shrinking': [True, False],
    'decision_function_shape': ['ovo', 'ovr']
}
gs_cv_sv = GridSearchCV(sv_clas, param_grid=parameters, scoring='accuracy', cv=5, n_jobs=4)
gs_cv_sv.fit(X, y)
gs_cv_sv.best_score_


# **Decision Tree**

# In[258]:


dt_clas = DecisionTreeClassifier(random_state=1)
selector = RFECV(dt_clas, cv=5)
selector.fit(X, y)
dt_cols = X.columns[selector.support_].values


# In[276]:


parameters = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [1, 2, 3]
}
gs_cv_dt = GridSearchCV(dt_clas, param_grid=parameters, scoring='accuracy', cv=5, n_jobs=4)
gs_cv_dt.fit(X[dt_cols], y)
gs_cv_dt.best_score_


# **Random Forest**

# In[280]:


rf_clas = RandomForestClassifier(random_state=1)
selector = RFECV(rf_clas, cv=5)
selector.fit(X, y)
rf_cols = X.columns[selector.support_].values


# In[301]:


parameters = {
    'n_estimators': [150, 180],
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [1, 2, 3]
}
gs_cv_rf = GridSearchCV(rf_clas, param_grid=parameters, scoring='accuracy', cv=5, n_jobs=4)
gs_cv_rf.fit(X[rf_cols], y)
gs_cv_rf.best_score_


# **Multi-layer Perceptron**

# In[310]:


mlp_clas = MLPClassifier(random_state=1, solver='lbfgs')
parameters = {
    'activation': ['identity', 'logistic', 'tanh', 'relu']
}
gs_cv_mlp = GridSearchCV(mlp_clas, param_grid=parameters, scoring='accuracy', cv=5, n_jobs=4)
gs_cv_mlp.fit(X, y)
gs_cv_mlp.best_score_


# So far **Random Forest** gave us the most accurate prediction. So we will use best parameters of our Random Forest model to train our entire train dataset.

# In[312]:


gs_cv_rf.best_params_


# In[367]:


rf_clas = RandomForestClassifier(random_state=1, criterion='entropy', max_depth=10, min_samples_leaf=2, min_samples_split=6, n_estimators=150)
rf_clas.fit(X[rf_cols], y)


# # Preparing the Test set

# In[368]:


test_set = pd.read_csv('../input/test.csv').drop(['Cabin', 'Ticket'], axis=1)
test_set.info()


# In[369]:


test_set['Fare'] = test_set['Fare'].fillna(test_set['Fare'].mean())


# In[370]:


test_set['Title'] = test_set['Name'].str.extract('(\w+(?=\.))', expand=False)
test_set = test_set.drop('Name', axis=1)
test_set.head()


# In[371]:


test_set['Title'] = test_set['Title'].str.replace('(Mlle|Mme|Ms)', 'Miss')
test_set['Title'] = test_set['Title'].str.replace('(Dr|Rev|Col|Major|Sir|Capt)', 'Occupation')
test_set['Title'] = test_set['Title'].str.replace('(Dona|Countess|Lady|Jonkheer)', 'Noble')
test_set['Title'].value_counts()


# In[372]:


test_set['FamilyCount'] = test_set['SibSp'] + test_set['Parch']
test_set['IsAlone'] = 0
test_set.loc[test_set['FamilyCount'] == 0, 'IsAlone'] = 1
test_set['IsMother'] = 0
test_set.loc[(test_set['Sex'] == 'female') & (test_set['Parch'] > 0) & (test_set['Title'] == 'Mrs'), 'IsMother'] = 1
test_set.head(10)


# In[373]:


test_set.loc[test_set['Sex'] == 'male', 'Sex'] = 0
test_set.loc[test_set['Sex'] == 'female', 'Sex'] = 1
test_set.head()


# In[374]:


test_set['FareBand'] = pd.cut(test_set['Fare'], bins=[0, 8, 15, 32, 100, 600], labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
test_set = test_set.drop('Fare', axis=1)
test_set = pd.concat([test_set, pd.get_dummies(test_set['Embarked'], drop_first=True), pd.get_dummies(test_set['Title'], drop_first=True)], axis=1).drop(['Embarked', 'Title'], axis=1)
test_set.head()


# In[375]:


test_set['FareBand'] = test_set['FareBand'].cat.codes
test_set.head()


# In[376]:


X_test_cols = ['Pclass', 'IsAlone', 'IsMother', 'Q', 'Miss', 'Mr', 'Mrs', 'Noble', 'Occupation']
test_set.loc[test_set['Age'].isnull(), 'Age'] = lin_reg_age.predict(test_set.loc[test_set['Age'].isnull(), X_test_cols])
test_set.head()


# In[377]:


test_set['AgeBand'] = pd.cut(test_set['Age'], bins=[0, 1, 12, 18, 21, 29, 36, 60, 100])
test_set = test_set.drop('Age', axis=1)
test_set.head()


# In[378]:


test_set['AgeBand'] = test_set['AgeBand'].cat.codes
test_set.head()


# # Prediction on Test set

# In[379]:


test_set['Survived'] = rf_clas.predict(test_set[rf_cols])
test_set.head()


# Let's make it ready for submission.

# In[ ]:


submission = test_set[['PassengerId', 'Survived']]
submission.to_csv('submission.csv', index=False)

