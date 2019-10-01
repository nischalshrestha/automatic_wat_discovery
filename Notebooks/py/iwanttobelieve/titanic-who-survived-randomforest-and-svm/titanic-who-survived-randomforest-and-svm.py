#!/usr/bin/env python
# coding: utf-8

# # Titanic - Who survived?
# I am new to data visualisation and machine learning, so any comments, any feedbacks, or any tips will be much appreciated.
# 
# ## Section 1. Data exploration
# ### 1.1. Importing libraries for data exploration and visualisation

# In[ ]:


# Data exploration and visualisation
import pandas as pd
pd.set_option('display.max_columns', 30) # avoiding truncated tables
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Statistics
from scipy.stats import chi2_contingency
from scipy.stats import chi2
import scipy.stats as stats


# ### 1.2. Reading in the data and first look at the data
# 
# There are **891 entries in the train dataset** and **418 entries in the test dataset**:
# * **Survived (train dataset only)**: the target variable showing 1 for passengers who survived and 0 for passengers who died,
# * **Pclass**: 3 travel classes coded as 1, 2 and 3,
# * **Name**: name of the passengers,
# * **Sex**: gender coded as a categorical variable (male and female), 
# * **Age**: continuous variable with 177 missing values in the train dataset and 86 in the test dataset. I am assuming age might be relevant to predicting the odds of surviving, so will look at filling in the missing data later on,
# * **SibSp**: number of siblings and spouses of the passengers,
# * **Parch**: number of parents and children of the passengers, can be combined with the SibSp variable to get the family size of the passenger,
# * **Ticket**: ticket number,
# * **Fare**: price paid by the passengers,
# * **Cabin**: there are too many missing data points; I will just probably not use this variable,
# * **Embarked**: only 2 missing values in the train dataset that we can probably easily impute.
# 
# Based on the train dataset, the **survival rate was 38%**. If we were to assume only that all the passengers died in a predictive model, the accuracy should be 62%.

# In[ ]:


# Datasets
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[ ]:


# Checking for missing data in the training dataset
print('Number of records:')
print(len(df_train.index))
print('{:=<70}'.format(''))
print('Missing values in the training dataset:')
print(df_train.isnull().sum())
print('{:=<70}'.format(''))


# In[ ]:


# Checking for missing data in the test dataset
print('Number of records:')
print(len(df_test.index))
print('{:=<70}'.format(''))
print('Missing values in the test dataset:')
print(df_test.isnull().sum())
print('{:=<70}'.format(''))


# In[ ]:


# Combining the two datasets
df_train['Dataset'] = 'train'
df_test['Dataset'] = 'test'
df_test.insert(loc=1, column='Survived', value=np.nan) # Insert as first column to be aligned with df_train
df_all = df_train.append(df_test)


# In[ ]:


# Checking for missing data in the test dataset
print('Number of records:')
print(len(df_all.index))
print('{:=<70}'.format(''))
print('Missing values in the test dataset:')
print(df_all.isnull().sum())
print('{:=<70}'.format(''))


# In[ ]:


# Percentage of passengers who survived (=38% of the total) 
print('Percentage of passengers who survived (1) and died (0):')
print(df_train['Survived'].value_counts(normalize=True))


# ### 1.3. Data exploration: gender and travel class seem to be strong predictors
# 
# * **Female had greater odds of survival**: not only did 74% of the females survive but females who survived were also over-represented compared to the females in the total population (35%),
# * **1st class travellers had greater odds of survival**: again, 63% of 1st travellers survived while they only represented less than a quarter (24%) of the total population,
# * **Male and 3rd class travellers had lesser odds of survival**: males and 3rd class travellers were under-represented among the passengers who survived (19% and 24% of the survivors, respectively) compared to the total population (65% and 55% of the total, respectively),
# * Looking at both gender and travel class, there is a higher proportion of female passengers who survived in the 1st (97%) and 2nd (92%) classes as opposed to the 3rd class (50%). By contrast, there was a lower proportion of survivors among male passengers regardless of the travel class.
# 
# Note that if we were to assume *only* that all the female passengers survived in a predictive model, the accuracy should be 74%. I guess we should be aiming at a score of at least 74% at the end.
# 

# In[ ]:


# Proportion of survivors based on gender (F=74%, M=19%)
summary_sex = pd.crosstab(index=df_train['Survived'],
                            columns=[df_train['Sex']],
                            margins=True)
                            
sex_survived = summary_sex / summary_sex.iloc[-1, :]

# Summary
print('{:=<70}'.format(''))
print('Proportion of survivors by gender:')
print(sex_survived)
print('{:=<70}'.format(''))

# Saving results for later use (visualisation)
f_survived = sex_survived.loc[1, 'female']
m_survived = sex_survived.loc[1, 'male']


# In[ ]:


# Proportion of survivors based on travel class (1st=63%, 2nd=47%, 3rd=24%)
summary_class = pd.crosstab(index=df_train['Survived'],
                            columns=[df_train['Pclass']],
                            margins=True)

pclass_survived = summary_class / summary_class.iloc[-1, :]

# Summary
print('{:=<70}'.format(''))
print('Proportion of survivors by travel class:')
print(pclass_survived)
print('{:=<70}'.format(''))

# Saving results for later use (visualisation)
c1_survived = pclass_survived.loc[1, 1]
c2_survived = pclass_survived.loc[1, 2]
c3_survived = pclass_survived.loc[1, 3]


# So far we've learned that there is a higher proportion of male (81%) and 3rd class passengers (76%) among the passengers who died. However, this is not enough to support the idea that they had lesser odds of surviving that the other passengers. After all, there was a higher proportion of male and 3rd class passengers among the total population and no one should have been surprised had the male population  represented 81% of the total population or had the 3rd class passengers represented 76% of the total population. More remarkable is the fact that male 3rd class passengers are under-represented in the population of survivors compared to their proportion in the total population. Now we are going to look at the over- or under-representation of survivors/dead passengers compared to theoriginal population.

# In[ ]:


# Proportion of passengers based on gender
sex = df_train['Sex'].value_counts(normalize=True)

# Summary
print('{:=<70}'.format(''))
print('Proportion of passengers by gender:')
print(sex)
print('{:=<70}'.format(''))

# Saving results for later use (visualisation)
f_passengers = sex['female']
m_passengers = sex['male']


# In[ ]:


# Proportion of passengers based on travel class
pclass = df_train['Pclass'].value_counts(normalize=True)

# Summary
print('{:=<70}'.format(''))
print('Proportion of passengers by travel class:')
print(pclass)
print('{:=<70}'.format(''))

# Saving results for later use (visualisation)
c1_passengers = pclass[1]
c2_passengers = pclass[2]
c3_passengers = pclass[3]


# In[ ]:


# Putting it all together (visualisation)
proportions_col = ['category', '%passengers', '%survived']
proportions_cat = ['female', 'male', 'class 1', 'class 2', 'class 3']
proportions_survived = [f_survived, m_survived, c1_survived, c2_survived, c3_survived]
proportions_passengers =[f_passengers, m_passengers, c1_passengers, c2_passengers, c3_passengers]

# Zipping passengers and survivors proportions into a dataframe
df_proportions = pd.DataFrame(
                            list(zip(proportions_cat,
                                    proportions_passengers,
                                    proportions_survived)
                                ),
                            columns=proportions_col)

df_proportions['variance'] = df_proportions['%passengers'] - df_proportions['%survived']

# Melting the dataframe in order to have the layout ready to display the data as a point plot
df_proportions_melted = df_proportions.melt(id_vars=['category', 'variance'],
                                            var_name='ratio type',
                                            value_name='ratio value')

# Dictionary showing categories with higher or lower odds of surviving. This will be used
# for the colour palette of the point plot.
my_red = '#EF6F6C'
my_green = '#435E53'
l_colours = [my_red if i >=0 else my_green for i in df_proportions['variance'].tolist()]
d_colours = dict(zip(df_proportions['category'],l_colours))

# Pointplot
fig, ax = plt.subplots(figsize=(6,8))
ax = sns.pointplot(x='ratio type', 
                    y='ratio value', 
                    hue='category', 
                    data=df_proportions_melted,
                    palette=d_colours)

# Add data labels as legend
for i in range(0, len(df_proportions)):
    lbl_offset = 1.1
    if (df_proportions['%passengers'] - df_proportions['%survived'])[i] < 0:
        ax.text(x=lbl_offset, 
                y=df_proportions['%survived'][i], 
                s=df_proportions['category'][i],
                ha='left',
                color=my_green)
    else:
        ax.text(x=lbl_offset, 
                y=df_proportions['%survived'][i], 
                s=df_proportions['category'][i],
                ha='left',
                color=my_red)

# Other bits of formatting
ax.set(title='Proportions of passengers in the total and survivors population',
        ylabel='Percentage')
ax.xaxis.label.set_visible(False)
ax.legend_.set_visible(False)
plt.show()


# In[ ]:


# Final summary table: there is a high proportion of surviving females in all travel classes but the 3rd (50%)
summary = pd.pivot_table(data=df_train,
                            index=['Survived'],
                            columns=['Sex', 'Pclass'],
                            values=['Name'],
                            aggfunc=('count'),
                            margins=True,
                            margins_name='Total')

print('{:=<70}'.format(''))
print(summary.div(summary.iloc[-1]))
print('{:=<70}'.format(''))


# In[ ]:


# Defining a function which will run a chi-square test of independence
def get_chi2(crosstab, HasMargin=True, proba=0.95):
    
    """
    Get a summary of a chi-square test of independence
    
    crosstab: frequency table in a Pandas crosstab format
    HasMargin: whether the crosstab has margins (totals) or not
    proba: maximum probability of accepting a false null hypothesis
    
    """
    
    # Data
    var1 = crosstab.index.name
    var2 = crosstab.columns.name
    
    # Integer offset to pass crosstab with no totals
    if HasMargin==True:
        i = -1
    else:
        i = ""
    
    # Contigency table from scipy
    stat, p, dof, expected = chi2_contingency(
                                            observed=crosstab.iloc[:i,:i],
                                            correction=True)
    
    # Independence of the variable: comparing chi2 result (stat) and the "critical" expected chi2 value
    # (which is based on the maximum probability of accepting a false null hypothesis and the degree of freedom)
    critical = chi2.ppf(proba, dof)
    if stat >= critical:
        chi2_independence = 'The variables {} and {} are dependent (X2>eX2)'.format(var1, var2)
    else:
        chi2_independence = 'The variables {} and {} are independent (X2<eX2)'.format(var1, var2)
    
    # Significance of the test: interpreting the p-value
    alpha = 1 - proba
    if p <= alpha:
        chi2_significance = 'The test is statistically significant (p<alpha)'
    else:
        chi2_significance = 'The test is not statistically significant (p>alpha)'

    # Summary of the results
    print('{:^70}'.format('Chi2 test of independence'))
    print('{:=<70}'.format(''))
    
    print('Tested variables')
    print(var1)
    print(var2)
    
    print('{:=<70}'.format(''))
    print('{}'.format('Test of independence'))
    print('Chi-square statistic (X2): {:.3f}'.format(stat))
    print('Minimum expected chi-square statistic (eX2): {:.3f}'.format(critical))
    print(chi2_independence)
        
    print('{:=<70}'.format(''))
    print('Significance of the test')
    print('p-value (p): {:.3f}'.format(p))
    print('Accepted significance level (alpha): {:.3f}'.format(alpha))
    print(chi2_significance)
    
    print('{:=<70}'.format(''))


# In[ ]:


# The chi-square test of independence suggests that there is a relationship between 'Survived' and 'Sex'
get_chi2(summary_sex, proba=0.99)


# In[ ]:


# The chi-square test of independence suggests that there is a relationship between 'Survived' and 'Pclass'
get_chi2(summary_class, proba=0.99)


# ### 1.4. Data exploration: single passengers seem to had lesser odds of survival
# In this section we are going to group 'Siblings/spouses' and 'Parents/children' into a single 'FamilySize' variable. Again, we are going to look at the over- or under-representation of the surviving populations compared to the total:
# * **Passengers travelling on their own showed lesser odds of surviving**: 30% of them survived while they represented 60% of the passengers,
# * **Passengers with a family had higher odds of surviving**.
# 

# In[ ]:


# Combining the SibSp and Parch variables into one 'FamilySize' variable
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1
df_all['FamilySize'] = df_all['SibSp'] + df_all['Parch'] + 1


# In[ ]:


# Proportion of passengers based on family size (and convert to dataframe for later merge)
family_passengers = df_train['FamilySize'].value_counts(normalize=True, sort=False).to_frame().reset_index()

# Renaming the columns of family_passengers
family_passengers.columns = ['FamilySize', '%total']

print('{:=<70}'.format(''))
print(family_passengers)
print('{:=<70}'.format(''))


# In[ ]:


# Proportion of survivors based on family size
summary_family = pd.crosstab(index=df_train['Survived'],
                            columns=[df_train['FamilySize']],
                            margins=True)

family_survived = summary_family / summary_family.iloc[-1, :]

# Only keeping a slice of the crosstab with the proportions of survivors (1) by family size, 
# excluding the totals ([:-1]) 
family_survived = family_survived.T[1][:-1].to_frame().reset_index()

# Renaming the columns of family_survived
family_survived.columns = ['FamilySize', '%survived']

print('{:=<70}'.format(''))
print(family_survived)
print('{:=<70}'.format(''))


# In[ ]:


# Merging family_passengers and family_survived into a single dataframe
df_family = pd.merge(left=family_passengers,
                     right=family_survived,
                     how='inner',
                     on='FamilySize')

# Melting the dataframe
df_family_melted = df_family.melt(id_vars=['FamilySize'],
                                    var_name='ratio type',
                                    value_name='ratio value')

# Panel charts showing the difference between proportions of in the total population and the population of
# survivors based on family size
ax = sns.catplot(data=df_family_melted,
                    x='ratio type',
                    y='ratio value',
                    col='FamilySize',
                    col_wrap=9,
                    kind='point',
                    height=3,
                    aspect=0.8)

# Formatting
ax.set_axis_labels('', 'Percentage')
plt.subplots_adjust(top=0.7)
plt.suptitle('Proportions of passengers: total and survivors')
plt.show()


# In[ ]:


# The chi-square test of independence suggests that there is a relationship between 'Survived' and 'FamilySize'
get_chi2(summary_family)


# ### 1.5. Data exploration: Is 'Age' really relevant to predicting survival?
# 
#  **Age seems to be mostly irrelevant to predicting survival**. The median age of passengers who survived and died is not massively different and shows a similar age distribution. On top of that, the point-biserial correlation of almost 0 sems to bear out that age is mostly irrelevant here.

# In[ ]:


# Age dataframe without missing values
df_age = df_train[['Age', 'Sex', 'Survived', 'Pclass']]
df_age = df_age.dropna(axis=0)

# Let's draw a swarmplot and a boxplot. Adding the boxplot onto the swarmplot shows a similar median age 
# between the two groups (survived and died) with a similar distribution.
fig, ax = plt.subplots(figsize=(8,8))
ax = sns.swarmplot(data=df_age, x='Survived', y='Age', hue='Sex')
ax = sns.boxplot(data=df_age, x='Survived', y='Age', orient='v', color='lightsteelblue')
ax.set(title='Age distribution among the population who survived and died')
plt.show()


# In[ ]:


# It looks like there is little to no correlation between age and surival/death, which is confirmed by a 
# Point-Biserial correlation of almost 0 (r=-0.08 with p=0.04).
stats.pointbiserialr(x=df_age['Survived'], y=df_age['Age'])


# In[ ]:


# First we are going to fill in the 177 missing 'Age' data points using the median age. Instead of just using 
# the median age, we will be using the median age by gender and travel class. I am not sure this will improve 
# anything later on but I thought it would be a good for me to learn something new.
#

# Median age by gender and by travel class
df_median_age = pd.pivot_table(data=df_age, 
                               index=['Sex','Pclass'],
                               values='Age',
                               aggfunc=np.median).reset_index()

# Function returning the median age depending on gender and travel class
def fill_age(data):
    
    median = df_median_age
    age = data['Age']
    sex = data['Sex']
    pclass = data['Pclass']
    
    if pd.isnull(age):
        
        if sex == 'female' and pclass == 1:
            return median[(median['Sex'] == 'female') & (median['Pclass'] == 1)]['Age'].values[0]
        if sex == 'female' and pclass == 2:
            return median[(median['Sex'] == 'female') & (median['Pclass'] == 2)]['Age'].values[0]
        if sex == 'female' and pclass == 3:
            return median[(median['Sex'] == 'female') & (median['Pclass'] == 3)]['Age'].values[0]
        if sex == 'male' and pclass == 1:
            return median[(median['Sex'] == 'male') & (median['Pclass'] == 1)]['Age'].values[0]
        if sex == 'male' and pclass == 2:
            return median[(median['Sex'] == 'male') & (median['Pclass'] == 2)]['Age'].values[0]
        if sex == 'male' and pclass == 3:
            return median[(median['Sex'] == 'male') & (median['Pclass'] == 3)]['Age'].values[0]
    
    else:
        
        return age

# Filling in the missing values
df_train['Age'] = df_train[['Age', 'Sex', 'Pclass']].apply(fill_age, axis=1)

# We use the data from the training dataset to fill in all the missing values in df_all to avoid data leakage
df_all['Age'] = df_all[['Age', 'Sex', 'Pclass']].apply(fill_age, axis=1)

# Check for null values
print('{:=<70}'.format(''))
print('Missing age values in the combined dataset:')
print(df_all['Age'].isnull().sum())
print('{:=<70}'.format(''))


# ### 5. Data exploration: port of embarkation
# 
# Passengers embarked either in Southampton ('S'), Cherbourg ('C) or Queenstown ('Q'). 
# 
# **The vast majority (73%) of the passengers embarked in Southampton but had a survival rate of 33%, or less than the overall 38% survival rate. By contrast, passengers who embarked in Cherbourg had a noticeably higher survival rate of 55%**. A reason for that might be that a higer proportion of passengers who embarked in Cherbourg travelled in first class (51%) as opposed to passengers who embarked in Southampton (20%).

# In[ ]:


# Proportion of passengers by port of embarkation
print(df_train['Embarked'].value_counts(normalize=True))


# In[ ]:


# The vast majority of the passengers embarked in Southampton. Let's just fill in the 2 missing
# piece of data with 'S'
df_train['Embarked'].fillna(value='S', inplace=True)
df_all['Embarked'].fillna(value='S', inplace=True)


# In[ ]:


# Survival rate based on the port of embarkation
summary_embarked = pd.crosstab(index=df_train['Survived'],
                                columns=df_train['Embarked'],
                                margins=True)

print(summary_embarked / summary_embarked.iloc[-1,:])


# In[ ]:


# Port of embarkation and travel class
port_class = pd.crosstab(index=df_train['Pclass'],
                                columns=df_train['Embarked'],
                                margins=True)

print(port_class / port_class.iloc[-1,:])


# In[ ]:


# The chi-square test of independence suggests that there is a relationship between 'Survived' and 'Embarked'
get_chi2(summary_embarked)


# ### 6. Data exploration: fares and adjusted fares
# The fares in the dataset are actually the fares paid for the corresponding ticket and not for the corresponding passenger alone. Several passengers share the same ticket (family or friends), which obviously gives an inaccurate picture of the individual fares. Therefore, we are going to calculate the average individual fare by passenger depending on the ticket number. Credit goes to [Geoffrey Wong](https://www.kaggle.com/csw4192/titanic-randomforest).
# Eventually, we find that survival based on the adjusted fares reflect survival based on travel classes. Therefore, we might not include the adjusted fares into our final model.

# In[ ]:


# Only one missing piece of data (test dataset)
df_all[df_all['Fare'].isnull() == True]


# In[ ]:


# Median fare paid by 3rd class travellers (train dataset)
median_fare = df_train['Fare'][df_train['Pclass']==3].median()

# Filling in the one missing fare (test dataset)
df_all['Fare'].fillna(value=median_fare, inplace=True)


# In[ ]:


# Showing a few examples of fares corresponding to a ticket and not a passengers
df_train[['Ticket', 'Fare', 'Name']].sort_values(by='Ticket', axis=0).head(5)


# In[ ]:


# Calculating adjusted fares in the combined dataset
d_ticket_count = dict(df_all['Ticket'].value_counts())
df_all['TicketCount'] = df_all['Ticket'].map(d_ticket_count)
df_all['AdjFare'] = df_all['Fare'] / df_all['TicketCount']

# Adding the adjusted fares in the training dataset
df_train['AdjFare'] = df_all[df_all['Dataset'] == 'train']['AdjFare']


# In[ ]:


# Fare and survival within each travel class
ax = sns.catplot(data=df_train,
                x='Survived',
                y='AdjFare',
                col='Pclass',
                col_wrap=3,
                kind='box',
                color='lightsteelblue',
                showfliers=False)


# ## Section 2. Features engineering
# 
# ### 2.1. Single passengers
# We saw that passengers travelling on their own (as defined by the number of family relationships) seem to have lesser odds of survival (see 1.4.). Now I would like to identify group of passengers travelling together based on their last name and ticket number, assuming that the fate of the different groups might have been similar.

# In[ ]:


# Extracting the last name of the passengers
df_all['LastName'] = df_all['Name'].str.extract(pat= '^([^,]*),', expand=True)
#df_all['LastName'] = df_all['Name'].str.split(pat= ',').str[0]


# In[ ]:


# Extracting only the digits from 'Ticket'
df_all['TicketNum'] = df_all['Ticket'].str.replace(pat= '(\D)', repl= '')
#df_all['TicketNum'] = df_all['Ticket'].str.extract(pat= '(\d+\d)', expand=True)


# In[ ]:


# Calculating adjusted fares in the combined dataset
d_ticket_count = dict(df_all['Ticket'].value_counts())
df_all['TicketCount'] = df_all['Ticket'].map(d_ticket_count)


# In[ ]:


# New 'SharedTicket' (0=False, 1=True)
df_all['SharedTicket'] = [1 if i > 1 else 0 for i in df_all['TicketCount']]

# New 'SharedName' (0=False, 1=True)
d_shared_name = dict(df_all['LastName'].value_counts())
df_all['SharedName'] = [1 if i > 1 else 0 for i in df_all['LastName'].map(d_shared_name)]  

# New 'SharedFeatures'
df_all['SharedFeatures'] = df_all['SharedTicket'] + df_all['SharedName']


# In[ ]:


# Adding 'IsSingle' column
shared_features = [
                    df_all['SharedFeatures'] == 0,
                    df_all['SharedFeatures'] == 1,
                    (df_all['SharedTicket'] == 0) & (df_all['SharedName'] == 1) & (df_all['FamilySize']  == 1),
                    (df_all['SharedTicket'] == 0) & (df_all['SharedName'] == 1) & (df_all['FamilySize']  > 1),
                    (df_all['SharedTicket'] == 1) & (df_all['SharedName'] == 0)
                  ]

is_single = [1, 0, 1, 0, 0]

    
df_all['IsSingle'] = np.select(shared_features, is_single)


# In[ ]:


df_all.sample(5)


# In[ ]:


# # Combining 'Ticket' and 'LastName'
# df_all['Group'] = df_all['LastName'] + df_all['TicketNum']

# # Get a list of all the unique groups
# group_keys = df_all['Group'].unique()

# # Get a dictionary of the unique groups with a unique incremental identifier (id: 'unique group')
# group_id = dict([(count, key) for (count, key) in enumerate(group_keys,start=1)])

# # Invert the keys and values ('unique group': id)
# group_id = dict([(v, k) for (k, v) in group_id.items()])

# # Adding 'GroupId' to the dataframe
# df_all['GroupId'] = df_all['Group'].map(group_id)


# ## Section 3. Predictions
# ### 3.1. Machine learning libraries

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# ### 3.2. Setting up the final dataframe

# In[ ]:


# Converting 'Sex' into a binary variable
df_all['Sex'] = [1 if i == 'female' else 0 for i in df_all['Sex']]

# Predictors
l_predictors = ['Dataset', 'Survived', 'Pclass', 'Sex', 'Embarked', 'IsSingle', 'FamilySize']

# Final dataframe
df_final = df_all[l_predictors]

# Get dummies
df_final = pd.get_dummies(data = df_final,
                          columns = ['Pclass', 'Embarked', 'FamilySize'])


# In[ ]:


df_final.sample(5)


# ### 3.3. Random Forest

# In[ ]:


# Going back to the training dataset to test our model
predictors = df_final[df_final['Dataset'] == 'train'].drop(['Dataset', 'Survived'], axis=1)
targets = df_final[df_final['Dataset'] == 'train']['Survived']
X_train, X_test, y_train, y_test = train_test_split(predictors, targets, test_size=0.30, random_state=0)


# In[ ]:


# Random forest
rfc = RandomForestClassifier(n_estimators=100, random_state = 42)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)

#print(classification_report(y_test,rfc_pred))
print(accuracy_score(rfc_pred, y_test))


# In[ ]:


# Tuning hyperparameters with GridSearchCV
param = {'n_estimators': [100, 500],
         'criterion' :['gini'],
         'max_features': ['auto'],
         'max_depth': [3, 4, 5]}
         
grid = GridSearchCV(estimator=rfc, param_grid=param, refit=True, cv=3)
grid.fit(X_train,y_train)
grid_pred = grid.predict(X_test)
print(accuracy_score(grid_pred, y_test))


# In[ ]:


# Best parameters from the grid search
grid.best_params_


# In[ ]:


# applying the best parameters to a new random forest model
rfc_best = RandomForestClassifier(random_state = 42,
                                  criterion = 'gini',
                                  max_depth = 4,
                                  max_features = 'auto',
                                  n_estimators = 100)

rfc_best.fit(X_train, y_train)
rfc_best_pred = rfc_best.predict(X_test)
print(accuracy_score(rfc_best_pred, y_test))


# ### 3.4. Support Vector Machine

# In[ ]:


# Support vector machine
svc = SVC(random_state=42, gamma='scale')
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)
print(accuracy_score(svc_pred, y_test))


# In[ ]:


# Tuning hyperparameters with GridSearchCV
param = {'C': [1, 10, 100, 1000],
         'gamma' :[1, 0.1, 0.001, 0.0001],
         'kernel': ['linear', 'rbf']}
         
svm_grid = GridSearchCV(estimator=svc, param_grid=param, refit=True, cv=3, iid=False)
svm_grid.fit(X_train,y_train)
svm_grid_pred = svm_grid.predict(X_test)
print(accuracy_score(svm_grid_pred, y_test))


# In[ ]:


# Best parameters from the grid search
svm_grid.best_params_


# In[ ]:


# Applying the best parameters to a new svm model
svm_best = SVC(random_state = 42,
                C = 1,
                gamma = 0.1,
                kernel = 'rbf')

svm_best.fit(X_train, y_train)
svm_best_pred = svm_best.predict(X_test)
print(accuracy_score(svm_best_pred, y_test))


# In[ ]:


# import thomas


# ### 3.5. File for submission

# In[ ]:


X_test_submit = df_final[df_final['Dataset'] == 'test'].drop(['Dataset', 'Survived'], axis=1)
submit_pred = rfc_best.predict(X_test_submit)

# File for submission
file_submit = pd.DataFrame({'PassengerId': df_test['PassengerId'].values,
                            'Survived': submit_pred.astype(np.int32)})

file_submit.to_csv('titanic_submit_pred.csv', index=False)

