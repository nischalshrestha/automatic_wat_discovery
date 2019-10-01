#!/usr/bin/env python
# coding: utf-8

# # What score can we expect using just the Name feature?
# 
# This notebook investigates working with the Titanic set using only the **Name** feature for prediction. 
# 
# This is motivated by an attempt to understand how each feature alone may contribute to the
# prediction of survival. Doing so for simple features such as gender (public score 0.76555) or passenger class (public score 0.6550) poses no problem and we know those scores are the optimum solutions under those restrictions. Other examples can be found [**here**](https://www.kaggle.com/pliptor/how-am-i-doing-with-my-score). Single feature prediction is not so straightforward for features such as **Name** or **Cabin** because or their more complex structure. 
# 
# Instead of the typical approach of finding *titles* in the Name field, identifying families and 
# manually creating engineered features, we will use a more "machine learning" automated approach of vectorizing the name field with the aid of sklearn.feature_extraction.text. By doing so, we will detach ourselves from the semantics of the words and use just word extraction and counting statistics. 
# 
# The Name field is initially expanded into a sparse 600+ dimensional space. We will demonstrate how to reduce the dimension of the problem using principal component analsys **(PCA)** to a more manageable 6-dimensional problem. This would also allow us to [**interactively visualize**](#3dplot) the Name feature in 3 dimensions! 
# 
# Finally, we will apply the **KNeighbors** algorithm solve it. We will optimize the KNeighbors parameter with GridSearch and also compute the cross validation to estimate the expected public score. Those familiar with this data set know that the gender can be extracted from the Name field. We therefore expect this approach to perform no worse than the gender-only approach. Are we going to succeed?
# 
# The picture below is a snapshot of the [interactive visualization](#3dplot) in section 2. You may use your mouse to rotate the structure and change its perspective.You'll be able to see clusters of survival and death that have been extracted just from the Name feature!
# 
# ![PCA](https://kaggle2.blob.core.windows.net/forum-message-attachments/281004/8555/newplot.png)
# 
# ## Contents
# 
# 1. [Reading data](#read_data)
# 2. [Vectorizing the Name feature and interactive 3D visualization](#vectorizing)
#     * [Using PCA to reduce dimension](#pca)
#     * [Interactive 3D Visualization](#3dplot)
# 3. [Modeling](#modeling)
# 4. [Predicting and creating a submission file](#submission)
# 
# [Conclusions](#conclusions)
# 
# 

# # 1) Reading data <a class="anchor" id="read_data"></a>
# 
# We will only read the relevant columns of the data. It will keep the data clean and prevent any accidental leakage of other features into our setup.

# In[ ]:


import pandas as pd
import numpy  as np

np.random.seed(2018)

# load data sets 
train = pd.read_csv('../input/train.csv', usecols =['Survived','PassengerId','Name'])
test  = pd.read_csv('../input/test.csv', usecols =['PassengerId','Name'])

# combine train and test for joint processing 
test['Survived'] = np.nan
comb = pd.concat([ train, test ])
comb.head()


# # 2) Vectorizing the Name feature and interactive 3D visualization <a class="anchor" id="vectorizing"></a>
# 
# The following steps vectorizes the Name feature using sklearn.feature_extraction.text. It transform a text 
# feature into numeric format.
# 
# We first create a pre-processing filter "clean_name" to remove punctuations out of the Name feature. 

# In[ ]:


# define a filter for the name field to remove punctuations and single letters
def clean_name(x):
    x = x.replace(',',' ').replace('.',' ').replace('(',' ').replace(')',' ').replace('"',' ').replace('-',' ')
    return ' '.join([ w for w in x.split() if len(w)> 1])


# ## 2.1) Defining the vectorizer
# 
# Two parameters will be passed to the vectorizer; the clean_name function above and min_df=2. What the latter does is 
# to make the vectorizer look at words that repeat at least twice in the entire data set. This way we remove all outliers that won't help with the statistics and will prevent model overfitting.

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
# setting punctuation filter and a minimum of terms to 2. 
count_vect = CountVectorizer(preprocessor=clean_name, min_df=2)


# ## 2.2) Fitting the vectorizer
# 
# It is fairly simple to fit the vectorizer into the Name feature! Note what it does is to build a dictionary of words (or tokens) and associate a unique number to it. Aaron corresponds to 0, Abbott to 1, etc. There are 638 distinct tokens. We print 6 terms in the dictionary for inspection.

# In[ ]:


# the following assigns a unique number for each word in the Name feature
count_vect.fit(comb['Name'])
for i,k in enumerate(count_vect.vocabulary_):
    if i>5:
        break
    else:
        print(k, count_vect.vocabulary_[k])


# ## 2.3) Transform the Name feature
# 
# Now we transform the Name feature to an array that indicates which word is included in each passenger's Name feature.

# In[ ]:


# the following transforms the Name feature to a vector indicating which word they contain
v = count_vect.transform(comb['Name'] )
va = np.array(v.toarray())
va.shape


# ## 2.4) Using PCA to reduce dimension <a class="anchor" id="pca"></a>
# 
# At this point we transformed the Name feature to a 638 dimension vector. While a previous version of this notebook posed no problem handling this size, we will demonstrate a dimension reduction technique using PCA to 6 dimensions. The reduced dimension will also alow us to inspect it in an interactive 3D plot (We will project only 3 dimensions out of 6 for this effect).

# In[ ]:


from sklearn.decomposition import PCA

#reducing va to 6 dimensions using PCA
pca = PCA(n_components = 6, random_state = 2018, whiten = True)
va = pca.fit_transform(va)

# we will plot only those that have a Survived label
va_survived = va[comb.index[comb.loc[:,'Survived']==1],...] 
va_perished = va[comb.index[comb.loc[:,'Survived']==0], ...]
va_na       = va[comb.index[comb.loc[:,'Survived']== np.nan],...]


# ## 2.5) 3D visualization <a class="anchor" id="3dplot"></a>
# We will use plotly for 3D visualization.

# In[ ]:


import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)

trace1 = go.Scatter3d(
    x=va_survived[:,1],
    y=va_survived[:,2],
    z=va_survived[:,3],
    mode='markers',
    marker=dict(
        size=12,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=1
    ),
    name = 'Survived'
)
trace2 = go.Scatter3d(
    x=va_perished[:,1],
    y=va_perished[:,2],
    z=va_perished[:,3],
    mode='markers',
    marker=dict(
        size=6,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=1
    ),
    name = 'Perished'
)
trace3 = go.Scatter3d(
    x=va_na[:,1],
    y=va_na[:,2],
    z=va_na[:,3],
    mode='markers',
    marker=dict(
        size=6,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=1
    ),
    name = 'Perished'
)
data = [trace1, trace2, trace3]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# Clusters of survial and death can be seen. Rembember this is all being done without any semantic intepretation of names or titles. It is was just done identifying unique terms and counting them. We do see that some points in different classes are almost overlapping.

# ## 2.6) Build a Name dataframe
# 
# Let's now build a dataframe from the above array and name each column from 0 to 5 (each PCA dimension after reduction).

# In[ ]:


# build a DataFrame from the array
feature_names = ['Nv' + str(i) for i in range(va.shape[1])]
NameVect = pd.DataFrame(data = va, index = comb.index, columns = feature_names)
NameVect.head()


# ## 2.7) Build df_train and df_test data frames
# 
# First we concatenate the survived target to the NameVect table and create a new dataframe

# In[ ]:


# comb2 now becomes the combined data in numeric form
comb2 = pd.concat([comb[['Survived']], NameVect],axis =1)
comb2.head()
comb2.to_csv('name_only_df.csv',index=False)


# Now we split back comb2 as we are done pre-processing

# In[ ]:


df_train = comb2.loc[comb2['Survived'].isin([np.nan]) == False]
df_test  = comb2.loc[comb2['Survived'].isin([np.nan]) == True]

print(df_train.shape)
df_train.head()


# In[ ]:


print(df_test.shape)
df_test.head()


# We are now ready for modeling!

# # 3) Modeling <a class="anchor" id="modeling"></a>
# 
# We will use a KNeighborsClassifier for the model and use GridSearchCV to tune it.

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knclass = KNeighborsClassifier(n_neighbors=11, metric = 'manhattan')
param_grid = ({'n_neighbors':[3,4,5,6,7,8,9,11,12,13],'metric':['manhattan','minkowski'],'p':[1,2]}) 
grs = GridSearchCV(knclass, param_grid, cv = 28, n_jobs=1, return_train_score = True, iid = False)
grs.fit(np.array(df_train[feature_names]), np.array(df_train['Survived']))


# Now that the tuning is completed, we print the best parameter found and also the estimated accuracy for the unseen data.  

# In[ ]:


print("Best parameters " + str(grs.best_params_))
gpd = pd.DataFrame(grs.cv_results_)
print("Estimated accuracy of this model for unseen data:{0:1.4f}".format(gpd['mean_test_score'][grs.best_index_]))


# # 4) Predicting and creating a submission file<a class="anchor" id="submission"></a>

# In[ ]:


pred_knn = grs.predict(np.array(df_test[feature_names]))

sub = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':pred_knn})
sub.to_csv('name_only_knn.csv', index = False, float_format='%1d')
sub.head()


# # Conclusions <a class="anchor" id="conclusions"></a>
# 
# We tackled the Titanic problem using only Name feature processing. The Name feature was transformed to a 600+ dimensional binary matrix using a count vectorizer in sklearn.feature_extraction.text. Next we reduced the dimension of the data matrix to 6 by using PCA. The prediction was then made with a KNneighbor optimizer tuned with a cross-validated grid search. As a by-product, we found the estimated accuracy for unseen data to be 0.7820. The obtained public score is 0.78468, which is higher than the gender-only optimal solution (0.76555) as we were hoping for!
# 
# Future work: Unfortunately we can't guarantee this solution for Name-only is optimal. It might be still possible to improve it both from the perspective of vectorization procedure and from KNneighbor optimization.
# I hope to do it at some time. 
# 
# Please let me know if you have questions or comments!
# 
