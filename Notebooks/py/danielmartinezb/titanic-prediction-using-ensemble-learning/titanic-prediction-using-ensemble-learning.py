#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This kernel is my first hands-on machine learning, any comments and suggestions are well received! (Sorry for grammar)


# # Table of contents
# * **Import required  libraries and helpful functions**
# * **Part One: Data Preparation**
#     * Load train dataset
#     * Dealing with missing values
#     * Feature analysis and transformations
# * **Part Two: Modeling**
#     * Fit and hyperparameter tuning of XGBoost classifier to the training set
#     * Fit and hyperparameter tuning of Random Forest classifier to the training set
#     * Fit and hyperparameter tuning of Supporting Vector Machine classifier to the training set
#     * Fit and hyperparameter tuning of an ensamble learning Voting classfier to the training set
#     * Cross-validation test
# * **Part Three: Predicting the Test Dataset**
#     * Load test dataset
#     * Dealing with missing values
#     * Feature transformations
#     * Creation of the csv file with predictions
#     

# # Import required libraries
# With some functions that I was constantly using throughout the code.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_missing_data_table(dataframe):
    '''Return the sum of missing values in dataframe and their percentage'''
    total = dataframe.isnull().sum()
    percentage = dataframe.isnull().sum() / dataframe.isnull().count()
    
    missing_data = pd.concat([total, percentage], axis='columns', keys=['TOTAL','PERCENTAGE'])
    return missing_data.sort_index(ascending=True)

def get_null_observations(dataframe, column):
    '''Return a DataFrame object with all rows with missing value in column'''
    return dataframe[pd.isnull(dataframe[column])]

def delete_null_observations(dataframe, column):
    '''Drop all rows of dataframe with missing value in column'''
    fixed_df = dataframe.drop(get_null_observations(dataframe,column).index)
    return fixed_df
    
def transform_dummy_variables(dataframe, columns):
    '''Return the One Hot encoding for all of the columns'''
    df = dataframe.copy()
    for column in columns:    
        df[column] = pd.Categorical(df[column])
    df = pd.get_dummies(df, drop_first=False)
    return df

def imput_nan_values(dataframe, column, strateg):
    from sklearn.preprocessing import Imputer
    imp = Imputer(strategy=strateg)
    df = dataframe.copy()
    df[column] = imp.fit_transform(df[column].values.reshape(-1,1))
    return df

print("Everything's ready!")


# # Part One: Data Preparation

# **Load train dataset**
# 

# In[ ]:


df = pd.read_csv('../input/train.csv')
df.head()


# After load the train dataset an initial approach is to describe the numerical features.

# In[ ]:


df.describe()


# The most relevant information here are the minimum and maximum values which show, for example, the range of people from babies up to elders (80 Y.O.). Also, the count value in 'Age' is a sign for possible missing values.

# **Dealing with missing values**
# 
# First let's visualize the sum of missing values for each feature and their percentage.

# In[ ]:


get_missing_data_table(df)


# To deal with those missing values I decided to exclude 'Cabin' due 77% missing values (high percentages like this usually mean something, a future job could be doing something about it).
# 
# The 2 rows where 'Embarked' is a missing value could be deleted or filled using the median, in this case I decided to exclude those rows for the training process.
# 
# The 19,87% missing values in age must be treated with care. I initially decided to fill those rows with arbitrary value '1000'.

# In[ ]:


# Delete Cabin
df = df.drop('Cabin', axis='columns') 

# Delete null observations in Embarked and reset the index in the DataFrame object
df = delete_null_observations(df, column='Embarked')
df = df.reset_index(drop=True)

# Fill missing values in 'Age' with arbitrary value 1000
df['Age'] = df['Age'].fillna(value=1000)

#Corroborate missing values elimination
get_missing_data_table(df)


# **Feature analysis and transformations** 
# 
# For this part I'll be using Tableau to perform a basic *ad-hoc* A/B test for the features, the idea is to describe the relation between each feature and the survival rate.
# 
# First, let's take a look at the survival rate.

# In[ ]:


get_ipython().run_cell_magic(u'HTML', u'', u"<div class='tableauPlaceholder' id='viz1531198007302' style='position: relative'><noscript><a href='#'><img alt='Survived ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ti&#47;TitanicsexABtest&#47;Survived&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='TitanicsexABtest&#47;Survived' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ti&#47;TitanicsexABtest&#47;Survived&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1531198007302');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# The chance of survival is 41% for a random selection. This will be the metric to determine if a feature affects the survival chance.
# 
# If a feature shows patterns that can increase or decrease that 41% then said feature is statistically significant. 
# 
# **Sex**

# In[ ]:


get_ipython().run_cell_magic(u'HTML', u'', u"<div class='tableauPlaceholder' id='viz1531199081048' style='position: relative'><noscript><a href='#'><img alt='Sex ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ti&#47;TitanicsexABtest&#47;Sex&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='TitanicsexABtest&#47;Sex' /><param name='tabs' value='no' /><param name='toolbar' value='no' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ti&#47;TitanicsexABtest&#47;Sex&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1531199081048');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# The plot shows that females have a higher survival chance than males (the reason is probably the rule of "women and children first "), so 'Sex' is definitly a feature to be used in the model.
# 
# **Fare**
# 
# To analyze 'Fare' I decided to create 3 groups of values : 
# * Less than 50
# * 50 to 100
# * More than 100

# In[ ]:


get_ipython().run_cell_magic(u'HTML', u'', u"<div class='tableauPlaceholder' id='viz1531629813892' style='position: relative'><noscript><a href='#'><img alt='Fare ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;7X&#47;7XN3CC9F8&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='path' value='shared&#47;7XN3CC9F8' /> <param name='toolbar' value='no' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;7X&#47;7XN3CC9F8&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1531629813892');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# The graph shows that a higher fare means a greater survival chance. 'Fare' is another variable to take into acount in the model.
# 
# **Pclass**

# In[ ]:


get_ipython().run_cell_magic(u'HTML', u'', u"<div class='tableauPlaceholder' id='viz1531359837278' style='position: relative'><noscript><a href='#'><img alt='Pclass ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ti&#47;TitanicsexABtest&#47;Pclass&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='TitanicsexABtest&#47;Pclass' /><param name='tabs' value='no' /><param name='toolbar' value='no' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ti&#47;TitanicsexABtest&#47;Pclass&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1531359837278');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# For 'Pclass' the graph shows that 1st and 2nd classes have a greater positive impact on the survival chance, so 'Pclass' will be included in the model.
# 
# **Embarked**

# In[ ]:


get_ipython().run_cell_magic(u'HTML', u'', u"<div class='tableauPlaceholder' id='viz1531361426048' style='position: relative'><noscript><a href='#'><img alt='Embarked ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ti&#47;TitanicsexABtest&#47;Embarked&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='TitanicsexABtest&#47;Embarked' /><param name='tabs' value='no' /><param name='toolbar' value='no' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ti&#47;TitanicsexABtest&#47;Embarked&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1531361426048');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# The graph shows a possitive impact for 'Cherbourg' but the other two are very close to the 41% line, I decided to use 'Pclass' in the model anyways but maybe 'Pclass' is not relevant enaugh and could be excluded from the model.
# 
# ** SibSp and Parch**

# In[ ]:


get_ipython().run_cell_magic(u'HTML', u'', u"<div class='tableauPlaceholder' id='viz1531361822427' style='position: relative'><noscript><a href='#'><img alt='SibSp ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ti&#47;TitanicsexABtest&#47;SibSp&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='TitanicsexABtest&#47;SibSp' /><param name='tabs' value='no' /><param name='toolbar' value='no' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ti&#47;TitanicsexABtest&#47;SibSp&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1531361822427');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>\n\n<div class='tableauPlaceholder' id='viz1531361843315' style='position: relative'><noscript><a href='#'><img alt='Parch ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ti&#47;TitanicsexABtest&#47;Parch&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='TitanicsexABtest&#47;Parch' /><param name='tabs' value='no' /><param name='toolbar' value='no' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ti&#47;TitanicsexABtest&#47;Parch&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1531361843315');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# 'SibSp' and 'Parch' are not very clear in the graph, the subclasses don't have an equal amount of records, and the ones in 'SibSp' are very close to the 41% line. I decided to exclude both from the model.
# 
# However to get value out of those features I decided to take the sum as a single feature called 'Family Size' the mainly reason is to answer if the size of the family affects the survival chance (a dissaster like Titanic looks worse if you are alone).

# In[ ]:


# family size = sibsp + parch
df['Family Size'] = df['SibSp'] + df['Parch']
df = df.drop('SibSp', axis='columns')
df = df.drop('Parch', axis='columns')
#Show first 5 records
df.head(5)


# In[ ]:


get_ipython().run_cell_magic(u'HTML', u'', u"<div class='tableauPlaceholder' id='viz1531362393780' style='position: relative'><noscript><a href='#'><img alt='Family Size ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ti&#47;TitanicsexABtest&#47;FamilySize&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='TitanicsexABtest&#47;FamilySize' /><param name='tabs' value='no' /><param name='toolbar' value='no' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ti&#47;TitanicsexABtest&#47;FamilySize&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1531362393780');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# According to the graph, having one to three family members means a greater survival chance than being alone. However, more than three reduce that chance. 'Family Size' is definitly a feature for the model.
# 
# **Name**
# 
# The feature 'Name' can't be tested with the type of graph used for the other variables, but each passenger has a unique name and by simple intuition, 'Name' is not a statistically singificant feature.
# 
# Nonetheless, the format that name is recorded in the dataset includes a title for each passenger, e.g. 'Mr' for the first record.

# In[ ]:


#Print the name of the first passenger
print(df['Name'][0])


# The 'Age' analysis showed that women are more likely to survive than men but title is a more complete category as the title is assigned according to the age of the passenger (important because of the missing 'Age' values to be filled in).
# 
# let's get the titles in the dataset and their distribution by age:

# In[ ]:


name_row = df['Name'].copy()
name_row = pd.DataFrame(name_row.str.split(', ',1).tolist(), columns = ['Last name', 'Name'])
name_row = name_row['Name'].copy()
name_row = pd.DataFrame(name_row.str.split('. ',1).tolist(),columns=["Title","Name"])
name_row = name_row['Title'].copy()

name_row.unique()


# In[ ]:


get_ipython().run_cell_magic(u'HTML', u'', u"<div class='tableauPlaceholder' id='viz1531365012539' style='position: relative'><noscript><a href='#'><img alt='Title Distribution ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ti&#47;TitanicsexABtest&#47;TitleDistribution&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='TitanicsexABtest&#47;TitleDistribution' /><param name='tabs' value='no' /><param name='toolbar' value='no' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ti&#47;TitanicsexABtest&#47;TitleDistribution&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1531365012539');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# I decided to create these title groups:
# * Master
# * Miss
# * Mr
# * Mrs
# * Others

# In[ ]:


get_ipython().run_cell_magic(u'HTML', u'', u"<div class='tableauPlaceholder' id='viz1531365774117' style='position: relative'><noscript><a href='#'><img alt='Title ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;PG&#47;PG9N22M7X&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='path' value='shared&#47;PG9N22M7X' /> <param name='toolbar' value='no' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;PG&#47;PG9N22M7X&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1531365774117');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# The difference between Mr vs Miss is greater than Female vs Male in 'Sex' because of the childrens recorded as Male. The feature 'Title' describe with more details the relationship between 'Sex' feature and the outcome 'Survived'. Title is statistically significant for the model.
# 
# Let's create this feature and drop 'Name':

# In[ ]:


#Add 'Title'
titles = name_row.tolist()
for i in range(len(titles)):
    title = titles[i]
    if title != 'Master' and title != 'Miss' and title != 'Mr' and title !='Mrs':
        titles[i] = 'Other'

name_row = pd.DataFrame(titles, columns=['Title'])
df['Title'] = name_row.copy()

#Drop 'Name'
df = df.drop('Name', axis='columns')
df.head(5)


# **Age**
# 
# To deal with missing values in 'Age' I decided to use the average age for the passenger's title.
# 
# The age distribution and average age by title is presented in the charts below:

# In[ ]:


get_ipython().run_cell_magic(u'HTML', u'', u"<div class='tableauPlaceholder' id='viz1531366447873' style='position: relative'><noscript><a href='#'><img alt='Title (group) Distribution ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ti&#47;TitanicsexABtest&#47;TitlegroupDistribution&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='TitanicsexABtest&#47;TitlegroupDistribution' /><param name='tabs' value='no' /><param name='toolbar' value='no' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ti&#47;TitanicsexABtest&#47;TitlegroupDistribution&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1531366447873');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>\n<br>\n<div class='tableauPlaceholder' id='viz1531367117802' style='position: relative'><noscript><a href='#'><img alt='Average age of each title group ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ti&#47;TitanicsexABtest&#47;Averageagebytitlegroup&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='TitanicsexABtest&#47;Averageagebytitlegroup' /><param name='tabs' value='no' /><param name='toolbar' value='no' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ti&#47;TitanicsexABtest&#47;Averageagebytitlegroup&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1531367117802');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# Let's fill the missing values according with those values:

# In[ ]:


test_df = df.copy()
test_df = pd.DataFrame([df['Age'].tolist(), df['Title'].tolist()]).transpose()
test_df.columns = ['Age','Title']

test_df_list = test_df.values #Age and Title of each row
for i in range(len(test_df_list)):
    age = test_df_list[i][0]
    title = test_df_list[i][1]
    
    if age == 1000: #Missing value
        if title == 'Master':
            test_df_list[i][0] = 5.19
        elif title == 'Miss':
            test_df_list[i][0] = 21.87
        elif title == 'Mr':
            test_df_list[i][0] = 32.18
        elif title == 'Mrs':
            test_df_list[i][0] = 35.48
        else:
            test_df_list[i][0] = 42.81

df['Age'] = test_df['Age'].copy() #Replace 'Age' in main DataFrame

#now Max value is not '1000'
df['Age'] = df['Age'].astype('float64')
df['Age'].describe()


# **Ticket and PassengerId**
# 
# The feature 'PassengerId' is unique for each passenger like 'Name' and it doesn't have a relation with 'Survived'. For 'Ticket' however the codification of the value could mean something useful for the model but for now I decided to exclude both variables from the model.
# 
# Let's drop those columns:

# In[ ]:


df = df.drop('Ticket', axis='columns')
df = df.drop('PassengerId', axis='columns')
#Show the actual structure
df.head(5)


# ** Dummy variables**
# 
# Now the DataFrame is almost complete, but features 'Sex', 'Pclass', 'Embarked' and 'Title' are categorical, let's transform those dummy variables using one of the functions defined at the beginning of this kernel:

# In[ ]:


#Treat categorical features
df = transform_dummy_variables(df,['Sex','Pclass','Embarked','Title'])
#Show changes
df.head(5)


# As I am going to use tree-based models for predictions it is not necessary to think about the 'Dummy Variables Trap' so the last thing to do is to split df into X and y. Scaling the values of the independent variables is also important:

# In[ ]:


#Getting X and y
X_train = df.iloc[:,1:].values
y = df.iloc[:,0].values

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

#Show changes
print('X_train: {0}'.format(X_train[0:5]))
print('y: {0}'.format(y[0:5]))


# # Part Two: Modeling
# 
# To build the model I decided to use tree-based algorithms because this is a prediction problem, the main goal is to get very accurate forecast, so the best decision is to use very high flexibity with low bias models like Support Vector Machines and Boosting.
# 
#  Kuhn and Johnson said in their book Applied Predictive Modeling that
# > “Unfortunately, the predictive models that are most powerful are usually the least interpretable.“ 
# 
# This trade-off between prediction accuracy and model interpretabilty is the reason why I decided to choose XGBoost, Random Forest and Supporting Vector Machine classifiers for the ensamble learning model.

# **Fit and hyperparameter tuning of XGBoost classifier to the training set**

# In[ ]:


from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y)


# For the hyperparameter tuning I decided to use random search:

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

folds = 4
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(classifier, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X_train,y), verbose=3, random_state=1001, iid=True)
random_search.fit(X_train, y)

xgboost_classifier = random_search.best_estimator_


# **Fit and hyperparameter tuning of Random Forest classifier to the training set**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y)


# Again Random search is used for hyperparameter tuning:

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
params = {
        'n_estimators': [5, 10, 15],
        'criterion': ['gini', 'entropy'],
        'max_features': ['auto', 'sqrt', 'log2', None],
        'max_depth': [None, 3, 4, 5]
        }

folds = 4
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(classifier, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X_train,y), verbose=3, random_state=1001, iid=True)
random_search.fit(X_train, y)

randomforest_classifier = random_search.best_estimator_


# **Fit and hyperparameter tuning of Supporting Vector Machine classifier to the training set**

# In[ ]:


from sklearn.svm import SVC
classifier = SVC(probability=True)
classifier.fit(X_train, y)


# Hyperparameter tuning with Random search:

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
params = {
        'C': [0.5, 1, 1.5],
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
        'gamma': [0.001, 0.0001],
        'class_weight': [None, 'balanced']
        }

folds = 4
param_comb = 5

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(classifier, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X_train,y), verbose=3, random_state=1001, iid=True)
random_search.fit(X_train, y)

svc_classifier = random_search.best_estimator_


# **Fit and hyperparameter tuning of an ensamble learning Voting classfier to the training set**
# 
# To get all the potential out of these three models I decided to use a soft Voting classifier due to the tuning of hyperparameters. I am confident that all three models are well calibrated so a prediction based on probabilities will be more efficient than a simple majority vote (hard voting).
# 
# let's implement the Voting classifier:

# In[ ]:


from sklearn.ensemble import VotingClassifier
classifier = VotingClassifier(estimators=[('xgb', xgboost_classifier), ('rf',randomforest_classifier), ('svc',svc_classifier)], voting='soft')
classifier.fit(X_train, y)


# **Cross-validation test**
# 
# Everything is ready to test the model, the way I decided to do it was through a cross-validation test with 5 folds:

# In[ ]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y, cv=5)
print('accuracy mean: {0}'.format(accuracies.mean()))
print('accuracy std: {0}'.format(accuracies.std()))


# The test results have an accuracy of more than 80% and a standard deviation of less than 3%, which I consider good for the problem.

# # Part Three: Predicting the Test Dataset

# **Load test dataset**

# In[ ]:


#Importing test dataset
df_test = pd.read_csv('../input/test.csv')
df_test.describe()


# **Dealing with missing values**
# The description of the test dataset shows missing values in 'Age' and 'Fare'. Let's see:

# In[ ]:


get_missing_data_table(df_test)


# This test dataset must be treated like the train dataset, 'Cabin' will be excluded as 'PassengerId', and 'Name, 'SibSp' and 'Parch' will be replaced by 'Family Size' and 'Title' will come up from 'Name'.
# 
# To deal with the missing values in 'Age' I will use the same method based on the passenger's title. The missing value in 'Fare' will be filled using the median.
# 
# Let's correct those missing values:

# In[ ]:


#Fill missing values in Fare using the median
df_test = imput_nan_values(df_test,'Fare','median')

#Fill missing values in Age according to the passanger's title
df_test['Age'] = df_test['Age'].fillna(value=1000)
name_row = df_test['Name'].copy()
name_row = pd.DataFrame(name_row.str.split(', ',1).tolist(), columns = ['Last name', 'Name'])
name_row = name_row['Name'].copy()
name_row = pd.DataFrame(name_row.str.split('. ',1).tolist(),columns=["Title","Name"])
name_row = name_row['Title'].copy()

titles = name_row.tolist()
for i in range(len(titles)):
    title = titles[i]
    if title != 'Master' and title != 'Miss' and title != 'Mr' and title !='Mrs':
        titles[i] = 'Other'

name_row = pd.DataFrame(titles, columns=['Title'])
df_test['Title'] = name_row.copy()

test_df = df_test.copy()
test_df = pd.DataFrame([df_test['Age'].tolist(), df_test['Title'].tolist()]).transpose()
test_df.columns = ['Age','Title']

test_df_list = test_df.values
for i in range(len(test_df_list)):
    age = test_df_list[i][0]
    title = test_df_list[i][1]
    
    if age == 1000:
        if title == 'Master':
            test_df_list[i][0] = 5.19
        elif title == 'Miss':
            test_df_list[i][0] = 21.87
        elif title == 'Mr':
            test_df_list[i][0] = 32.18
        elif title == 'Mrs':
            test_df_list[i][0] = 35.48
        else:
            test_df_list[i][0] = 42.81

df_test['Age'] = test_df['Age'].copy()

#Show changes
get_missing_data_table(df_test)


# **Feature transformations**
# 
# Let's make the other feature transfomations:

# In[ ]:


#Drop Cabin
df_test = df_test.drop('Cabin', axis='columns')

#Create Family Size = Sibsp + Parch
df_test['Family Size'] = df_test['SibSp'] + df_test['Parch']
df_test = df_test.drop('SibSp', axis='columns')
df_test = df_test.drop('Parch', axis='columns')

#Drop irrelevant features
df_test = df_test.drop('Name', axis='columns')
df_test = df_test.drop('Ticket', axis='columns')
df_test = df_test.drop('PassengerId', axis='columns')

#Transform dummy variables
df_test['Age'] = df_test['Age'].astype('float64')
df_test = transform_dummy_variables(df_test,['Sex','Pclass','Embarked','Title'])

#Show changes
df_test.head(5)


# **Creation of the csv file with predictions**

# In[ ]:


#Predictions
X_test = df_test.values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_test = sc.fit_transform(X_test)

pred = classifier.predict(X_test)

# Create result dataframe and 'predictions.csv'
test_dataset = pd.read_csv('../input/test.csv')
ps_id = test_dataset.iloc[:,0].values
d = {'PassengerId':ps_id, 'Survived':pred}
df = pd.DataFrame(data=d)
df = df.set_index('PassengerId')
df.to_csv('predictions.csv')

#Show structure
df.head(15)


# ![score](https://i.imgur.com/Q3I02Zy.png)
