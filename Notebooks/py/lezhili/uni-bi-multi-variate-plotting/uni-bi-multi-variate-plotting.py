#!/usr/bin/env python
# coding: utf-8

# # Univariate, Bivariate, Multivariate Plotting
# 
# <table>
# <tr>
# <td><img src="https://i.imgur.com/skaZPhb.png" width="350px"/></td>
# <td><img src="https://i.imgur.com/gaNttYd.png" width="350px"/></td>
# <td><img src="https://i.imgur.com/pampioh.png"/></td>
# <td><img src="https://i.imgur.com/OSbuszd.png"/></td>
# 
# <!--<td><img src="https://i.imgur.com/ydaMhT1.png" width="350px"/></td>
# <td><img src="https://i.imgur.com/WLAqDSV.png" width="350px"/></td>
# <td><img src="https://i.imgur.com/Tj2y9gH.png"/></td>
# <td><img src="https://i.imgur.com/X0qXLCu.png"/></td>-->
# </tr>
# <tr>
# <td style="font-weight:bold; font-size:16px;">Bar Chat</td>
# <td style="font-weight:bold; font-size:16px;">Line Chart</td>
# <td style="font-weight:bold; font-size:16px;">Area Chart</td>
# <td style="font-weight:bold; font-size:16px;">Histogram</td>
# </tr>
# <tr>
# <td>df.plot.bar()</td>
# <td>df.plot.line()</td>
# <td>df.plot.area()</td>
# <td>df.plot.hist()</td>
# </tr>
# <tr>
# <td>Good for nominal and small ordinal categorical data.</td>
# <td>	Good for ordinal categorical and interval data.</td>
# <td>Good for ordinal categorical and interval data.</td>
# <td>Good for interval data.</td>
# </tr>
# <tr>
#     <tr>
# <td><img src="https://i.imgur.com/bBj1G1v.png" width="350px"/></td>
# <td><img src="https://i.imgur.com/ChK9zR3.png" width="350px"/></td>
# <td><img src="https://i.imgur.com/KBloVHe.png" width="350px"/></td>
# <td><img src="https://i.imgur.com/C7kEWq7.png" width="350px"/></td>
# </tr>
# <tr>
# <td style="font-weight:bold; font-size:16px;">Scatter Plot</td>
# <td style="font-weight:bold; font-size:16px;">Hex Plot</td>
# <td style="font-weight:bold; font-size:16px;">Stacked Bar Chart</td>
# <td style="font-weight:bold; font-size:16px;">Bivariate Line Chart</td>
# </tr>
# <tr>
# <td>df.plot.scatter()</td>
# <td>df.plot.hexbin()</td>
# <td>df.plot.bar(stacked=True)</td>
# <td>df.plot.line()</td>
# </tr>
# <tr>
# <td>Good for interval and some nominal categorical data.</td>
# <td>Good for interval and some nominal categorical data.</td>
# <td>Good for nominal and ordinal categorical data.</td>
# <td>Good for ordinal categorical and interval data.</td>
# </tr>
# <tr>
# <td><img src="https://i.imgur.com/gJ65O47.png" width="350px"/></td>
# <td><img src="https://i.imgur.com/3qEqPoD.png" width="350px"/></td>
# <td><img src="https://i.imgur.com/1fmV4M2.png" width="350px"/></td>
# <td><img src="https://i.imgur.com/H20s88a.png" width="350px"/></td>
# </tr>
# <tr>
# <td style="font-weight:bold; font-size:16px;">Multivariate Scatter Plot</td>
# <td style="font-weight:bold; font-size:16px;">Grouped Box Plot</td>
# <td style="font-weight:bold; font-size:16px;">Heatmap</td>
# <td style="font-weight:bold; font-size:16px;">Parallel Coordinates</td>
# </tr>
# <tr>
# <td>df.plot.scatter()</td>
# <td>df.plot.box()</td>
# <td>sns.heatmap</td>
# <td>pd.plotting.parallel_coordinates</td>
# </tr>
# </table>
# 
# ----
# 
# We will be using [Pandas](https://pandas.pydata.org/) for loading, transforming and analyzing our data. 

# In[ ]:


# kernel adapted from https://www.kaggle.com/residentmario/welcome-to-data-visualization


# In[ ]:


# loading data
import pandas as pd
reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)
reviews.head(3)


# # Univariate Charts
# ## Bar charts and categorical data
# 
# Bar charts are arguably the simplest data visualization. They map categories to numbers: the amount of eggs consumed for breakfast (a category) to a number breakfast-eating Americans, for example; or, in our case, wine-producing provinces of the world (category) to the number of labels of wines they produce (number):

# In[ ]:


reviews['province'].value_counts().head(10).plot.bar()


# In[ ]:


# percentage instead of raw numbers
(reviews['province'].value_counts().head(10) / len(reviews)).plot.bar()


# In[ ]:


# what about using bar chart for numeric feature
reviews['points'].value_counts().sort_index().plot.bar()


# ## Line charts
# 
# The wine review scorecard has 20 different unique values to fill, for which our bar chart is just barely enough. What would we do if the magazine rated things 0-100? We'd have 100 different categories; simply too many to fit a bar in for each one!
# 
# In that case, instead of bar chart, we could use a line chart:

# In[ ]:


reviews['points'].value_counts().sort_index().plot.line()


# ## Area charts
# 
# Area charts are just line charts, but with the bottom shaded in. That's it!

# In[ ]:


reviews['points'].value_counts().sort_index().plot.area()


# ## Histograms

# In[ ]:


reviews[reviews['price'] < 200]['price'].plot.hist()


# A histogram looks, trivially, like a bar plot. And it basically is! In fact, a histogram is special kind of bar plot that splits your data into even intervals and displays how many rows are in each interval with bars. The only analytical difference is that instead of each bar representing a single value, it represents a range of values.
# 
# However, histograms have one major shortcoming (the reason for our 200$ caveat earlier). Because they break space up into even intervals, they don't deal very well with skewed data:

# In[ ]:


reviews['price'].plot.hist()


# This is the real reason I excluded the >$200 bottles earlier; some of these vintages are really expensive! And the chart will "grow" to include them, to the detriment of the rest of the data being shown.

# In[ ]:


reviews[reviews['price'] > 1500]


# There are many ways of dealing with the skewed data problem; those are outside the scope of this tutorial. The easiest is to just do what I did: cut things off at a sensible level.
# 
# This phenomenon is known (statistically) as **skew**, and it's a fairly common occurance among interval variables.
# 
# Histograms work best for interval variables without skew. They also work really well for ordinal categorical variables like `points`:

# In[ ]:


reviews['points'].plot.hist()


# ## Scatter plot
# 
# The simplest bivariate plot is the lowly **scatter plot**. A simple scatter plot simply maps each variable of interest to a point in two-dimensional space. This is the result:

# In[ ]:


reviews[reviews['price'] < 100].sample(100).plot.scatter(x='price', y='points')


# In[ ]:


# when number of data points gets larger
reviews[reviews['price'] < 100].plot.scatter(x='price', y='points')


# ## Hexplot
# 
# A  **hex plot** aggregates points in space into hexagons, and then colors those hexagons based on the values within them:

# In[ ]:


reviews[reviews['price'] < 100].plot.hexbin(x='price', y='points', gridsize=15)


# ## Stacked plots
# 
# Scatter plots and hex plots are new. But we can also use the simpler plots we saw in the last notebook.
# 
# The easiest way to modify them to support another visual variable is by using stacking. A stacked chart is one which plots the variables one on top of the other.
# 
# We'll use a supplemental selection of the five most common wines for this next section.

# In[ ]:


wine_counts = pd.read_csv("../input/most-common-wine-scores/top-five-wine-score-counts.csv",
                          index_col=0)
wine_counts.head()


# In[ ]:


wine_counts.plot.bar(stacked=True)


# In[ ]:


wine_counts.plot.area()


# ## Bivariate line chart
# 
# One plot type we've seen already that remains highly effective when made bivariate is the line chart. Because the line in this chart takes up so little visual space, it's really easy and effective to overplot multiple lines on the same chart.

# In[ ]:


wine_counts.plot.line()


# # Multivariate Charts

# In[ ]:


import pandas as pd
pd.set_option('max_columns', None)
df = pd.read_csv("../input/fifa-18-demo-player-dataset/CompleteDataset.csv", index_col=0)

import re
import numpy as np

footballers = df.copy()
footballers['Unit'] = df['Value'].str[-1]
footballers['Value (M)'] = np.where(footballers['Unit'] == '0', 0, 
                                    footballers['Value'].str[1:-1].replace(r'[a-zA-Z]',''))
footballers['Value (M)'] = footballers['Value (M)'].astype(float)
footballers['Value (M)'] = np.where(footballers['Unit'] == 'M', 
                                    footballers['Value (M)'], 
                                    footballers['Value (M)']/1000)
footballers = footballers.assign(Value=footballers['Value (M)'],
                                 Position=footballers['Preferred Positions'].str.split().str[0])


# In[ ]:


footballers.head()


# ## Multivariate scatter plots
# 
# Supose that we are interested in seeing which type of offensive players tends to get paid the most: the striker, the right-winger, or the left-winger.

# In[ ]:


import seaborn as sns

sns.lmplot(x='Value', y='Overall', hue='Position', 
           data=footballers.loc[footballers['Position'].isin(['ST', 'RW', 'LW'])], 
           fit_reg=False)


# In[ ]:


sns.lmplot(x='Value', y='Overall', markers=['o', 'x', '*'], hue='Position',
           data=footballers.loc[footballers['Position'].isin(['ST', 'RW', 'LW'])],
           fit_reg=False
          )


# ### Grouped box plot
# 
# Another demonstrative plot is the grouped box plot. This plot takes advantage of **grouping**. Suppose we're interested in the following question: do Strikers score higher on "Aggression" than Goalkeepers do?

# In[ ]:


f = (footballers
         .loc[footballers['Position'].isin(['ST', 'GK'])]
         .loc[:, ['Value', 'Overall', 'Aggression', 'Position']]
    )
f = f[f["Overall"] >= 80]
f = f[f["Overall"] < 85]
f['Aggression'] = f['Aggression'].astype(float)

sns.boxplot(x="Overall", y="Aggression", hue='Position', data=f)


# ### Heatmap
# 
# Probably the most heavily used summarization visualization is the **correlation plot**, in which measures the correlation between every pair of values in a dataset and plots a result in color.

# In[ ]:


f = (
    footballers.loc[:, ['Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control']]
        .applymap(lambda v: int(v) if str.isdecimal(v) else np.nan)
        .dropna()
).corr()

sns.heatmap(f, annot=True)


# ### Parallel Coordinates
# 
# A **parallel coordinates plot** provides another way of visualizing data across many variables.

# In[ ]:


from pandas.plotting import parallel_coordinates

f = (
    footballers.iloc[:, 12:17]
        .loc[footballers['Position'].isin(['ST', 'GK'])]
        .applymap(lambda v: int(v) if str.isdecimal(v) else np.nan)
        .dropna()
)
f['Position'] = footballers['Position']
f = f.sample(200)

parallel_coordinates(f, 'Position')


# # Now it's your turn!
# 
# We've learned about the handful of different kinds of data, and looked at some of the tools for plotting them.
# Now, let make use of the technologies to do **exploratory data analysis** (EDA) on a different dataset. We'll be working with the Titanic dataset.
# 
# ## Task 1: finding insights in the Titanic dataset using visualization techniques we introduced above (25 min)
# ---
# 
# ***Plotting libraries documentations for your inferrence: ***
# - [Pandas](https://pandas.pydata.org/pandas-docs/version/0.22/api.html#api-dataframe-plotting)
# - [Seaborn](https://seaborn.pydata.org/index.html)

# In[ ]:


# import dataset 
titanic = pd.read_csv("../input/titanic/train.csv")
titanic.head(10)


# 
# ## Task 2: sharing your insights with classmates (15 min)

# In[ ]:




