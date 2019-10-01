#!/usr/bin/env python
# coding: utf-8

# ## Seaborn Tutorial (basic)

# ###### In this tutorial we will progress through seaborn in the sequence presented in the Seaborn Documentation.
# Note: All properties within  a particular visualization might not be explored. 
# For a list of all properties kindly visit: https://seaborn.pydata.org/

# In[ ]:


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# We can list datasets available within seaborn using
# sns.get_dataset_names().
# 
# For this tutorial we will be using 2 datasets:
# 1. Iris 
# 2. Titanic

# In[ ]:


iris = sns.load_dataset('iris')
titanic = sns.load_dataset('titanic')


# In[ ]:


# preview of the iris dataset
iris.head()


# ### Visualizing statistical relationships

# #### Relating variables with scatter plots

# In[ ]:


# lets try to visualize features sepal length & width to better understand the data
sns.relplot(x= 'sepal_length', y= 'sepal_width', data=iris)


# The above graph does not differentiate between different classifications, hence it is not much helpful

# In[ ]:


# let us add another dimension - the species dimension using 'hue'
sns.relplot(x= 'sepal_length', y='sepal_width', hue='species', data=iris)


# The above graph emphasises the difference between the classes.
# 
# We can also achieve the above effect using a 'size' feature, this will make each classification a different size

# In[ ]:


# visualizing petal length & width
sns.relplot(x= 'petal_length', y='petal_width', size='species', data=iris)


# #### Emphasizing continuity with line plots

# In[ ]:


# creating a time series dataset
time = pd.DataFrame(dict(time = np.arange(500), value = np.random.randn(500).cumsum()))


# In[ ]:


# plotting the time series using a line chart
sns.relplot(x= 'time', y= 'value', kind= 'line', data= time)


# In[ ]:


# Preview of the titanic dataset
titanic.head()


# In[ ]:


# when there are multiple measurements for the same value 
# ex. in the Titanic database- mulitple 'fare' values for the same 'class'
# seaorn creates a confidence interval to represent them
sns.relplot(x= 'pclass', y= 'fare', kind= 'line', data = titanic)


# In[ ]:


# The CI (confidence interval) feature can be turned off using: None
# It can be changed to standard deviation to represent the distribution at each classification using: 'sd'

sns.relplot(x= 'pclass', y= 'fare', kind= 'line', ci= None, data = titanic)


# #### Plotting subsets of data 

# In[ ]:


# segregating the fare v. pclass according to sex of the passenger
sns.relplot(x= 'pclass', y= 'fare', kind= 'line', hue='sex', data= titanic)


# We observe that fare for females was more than that for males, regardless of class

# In[ ]:


# we can add another variable in this if required using 'event' property
sns.relplot(x= 'pclass', y= 'fare', kind= 'line', hue='sex', style= 'survived', ci= None, data= titanic)


# In[ ]:


# Changing the color palette
# For a comprehensive list of color palettes, kindly visit the documentation. Link provided at the beginning
palette = sns.cubehelix_palette(light = .7, n_colors= 2)
sns.relplot(x= 'pclass', y= 'fare', kind= 'line', hue='sex', style= 'survived', palette= palette, ci= None, data= titanic)


# ###### Exploring multi plot graphs

# In[ ]:


# We use the 'col' feature to create a multi plot graph
# Here each graph/column represents a different classification and the 'x' and 'y' are plotted for each 
sns.relplot(x= 'age', y= 'fare', col='survived', data= titanic)


# In[ ]:


# we can add additional features, as we did earlier
sns.relplot(x= 'age', y= 'fare', col='survived', hue= 'sex', data= titanic)


# ## Plotting with categorical data

# ### Categorical scatter plots

# In[ ]:


# Default plot with catplot is scatterplot
# This helps in visualizing categorical variables
sns.catplot(x= 'sex', y= 'age', data= titanic)


# In[ ]:


# We can control the magnitude of jitter using the 'jitter' feature
sns.catplot(x= 'sex', y= 'age', jitter= False, data= titanic)


# In[ ]:


# For small datasets, we can check the distribution of the data using 'swarm' plot
sns.catplot(x= 'sex', y= 'age', kind= 'swarm', data= titanic)


# We can observe in the above graph that majority of the males were in the age group of 20 - 40

# In[ ]:


# we can further add more features in the plot using options like 'hue'
sns.catplot(x= 'survived', y= 'age', kind= 'swarm', hue= 'sex', data= titanic)


# As the above graph depicts, survival rate was higher for females

# In[ ]:


# visualizing a subset of the data
sns.catplot(x= 'survived', y= 'age', kind= 'swarm', hue= 'sex', data= titanic.query('pclass==1'))


# We have added a criteria in the plot above- to only display data for PClass = 1. Thus, we can create plots for specific subset of data

# ### Distribution of observations within categories

# #### Boxplots

# In[ ]:


# Age wise distribution in each class
sns.catplot(x= 'pclass', y= 'age', kind= 'box', data= titanic)


# In[ ]:


# Adding additional feature 
sns.catplot(x= 'pclass', y= 'age', hue= 'sex', kind= 'box', data= titanic)


# In[ ]:


# A better distribution plot is boxen plot for larger datasets, it sshows the shape of distribution as well
sns.catplot(x= 'pclass', y= 'age', kind= 'boxen', data= titanic)


# In[ ]:


# A violin plot provides distribution along with IQR plot embedded in it
sns.catplot(x= 'pclass', y= 'age', kind= 'violin', data= titanic)


# In[ ]:


# We can combine swarm and violin plot to show individual points in the distribution
g = sns.catplot(x= 'pclass', y= 'age', kind= 'violin', inner= None, data= titanic)
sns.swarmplot(x= 'pclass', y= 'age', color= 'k', data= titanic, ax= g.ax)


# ### Statistical estimation within categories

# #### Bar plots

# In[ ]:


sns.catplot(x="sex", y="survived", hue="class", kind="bar", data=titanic)


# In[ ]:


# When we want to show the number of observations in each category without creating a quantitative variable
sns.catplot(x= 'deck', kind= 'count', palette= 'ch:.25', data= titanic)


# As we can observe the number of people on deck C were the highest

# In[ ]:


# Adding more variables in our analysis
sns.catplot(x= 'class', kind= 'count', hue='sex', palette= 'pastel', data= titanic)


# As we can observe, the number of male, female in PClass 1 and 2 were approximately equal, whereas the number of males were substantially larger than the number of females in PClass 3

# #### Point plots

# In[ ]:


sns.catplot(x= 'sex', y= 'survived', hue= 'class', kind= 'point', data= titanic)


# #### Showing multiple relationships through multi-plots

# In[ ]:


sns.catplot(x= 'class', y= 'age', hue='sex', col= 'survived', kind= 'swarm', data= titanic)


# As we observe, female survival rate was higher and female survival rate among PClass 1 and 2 relatively higher than PClass 3

# ## Visualizing the distribution of a dataset

# ### Plotting univariate distributions

# In[ ]:


# Creating a variable with gaussian distribution
x= np.random.normal(size= 100)
# Plotting a histogram and a kernel density estimate
sns.distplot(x)


# #### Histograms

# In[ ]:


# Plotting a histogram along with a small vertical tick at each observation
sns.distplot(x, bins=20, kde= False, rug= True)


# ## Building structured multi-plot grids

# We have been creating multi plot grids in a way through the 'col' property in relplot and catplot. This is done as each of them uses the FacetGrid object internally.

# In[ ]:


# In multi dimensional data, a useful approach is to draw multiple instances of the same plot on different subsets of the dataset
# The followng command is used to initialize the FacetGrid object with dataframe and row, column, hue 
g= sns.FacetGrid(titanic, col= 'survived', hue= 'sex')
# Adding the kind of plot and features to visualize using the 'map' function
g.map(plt.scatter, 'deck', 'age')
g.add_legend()


# In[ ]:


# We can change features like height and aspect to alter the look and feel of the plot
g= sns.FacetGrid(titanic, col= 'survived', height= 5, aspect= .6)
g.map(sns.barplot, 'sex', 'fare')


# ## Plotting pairwise data relationships

# ###### Pair plot gives a relationship between each pair of the available features

# In[ ]:


g= sns.PairGrid(iris)
g.map(plt.scatter)


# In[ ]:


# We can add other features and properties to enhance the visualization
g= sns.PairGrid(iris, hue= 'species', height= 3, aspect= 0.7)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend()


# ##### Hope this was useful :)
