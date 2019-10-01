#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import Required Library
import numpy as np
import pandas as pd
# for Chart
from bokeh.plotting import figure
from bokeh.io import output_notebook, show
from bokeh.models import ColumnDataSource, FactorRange, ranges, LabelSet

output_notebook()


# In[ ]:


# Read Data
df = pd.read_csv("../input/train.csv") 


# In[ ]:


# Clean Bad Values
df = df.dropna() 


# In[ ]:


df.head()


# In[ ]:


surv_counts = df.Survived.value_counts()
source = ColumnDataSource(dict(x=["Died", "Survived"],y=surv_counts.values))

plot = figure(plot_width=300, plot_height=300, title="Total Survived Stat", x_range = source.data["x"])

plot.vbar(source=source, x='x', width=0.5, top='y', name='y')
show(plot)


# In[ ]:


surv_counts = df.Survived[df.Sex == 'female'].value_counts()
source = ColumnDataSource(dict(x=["Died", "Survived"],y=surv_counts.values))

plot = figure(plot_width=300, plot_height=300, title="Total Female Survived Stat", y_axis_label = "count", x_range = source.data["x"])

plot.vbar(source=source, width=0.5, x='x', top='y')

show(plot)


# In[ ]:


surv_counts = df.Survived[df.Sex == 'male'].value_counts()
source = ColumnDataSource(dict(x=["Died", "Survived"],y=surv_counts.values))

plot = figure(plot_width=300, plot_height=300, title="Total Male Survived Stat", y_axis_label = "count", x_range = source.data["x"])

plot.vbar(source=source, width=0.5, x='x', top='y')

show(plot)


# In[ ]:


gender_counts = df.Sex[df.Survived == 1].value_counts()
source = ColumnDataSource(dict(x=gender_counts.keys().tolist(),y=gender_counts.values))

plot = figure(plot_width=300, plot_height=300, title="Gender Survived Stat", y_axis_label = "count", x_range = source.data["x"])

plot.vbar(source=source, width=0.5, x='x', top='y')

show(plot)


# In[ ]:


# Learning Add Simple Model
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestRegressor

str_cols = df.columns[df.columns.str.contains('Sex')]
clfs = {c:LabelEncoder() for c in str_cols}
for col, clf in clfs.items():
    df[col] = clfs[col].fit_transform(df[col])

features = ['Pclass', 'Sex', 'SibSp', 'Parch']
for_model = df[features]

titanic_model = RandomForestRegressor(random_state=1)
y = df.Survived
titanic_model.fit(for_model, y)


# In[ ]:


test_data = pd.read_csv('../input/test.csv')
test_data = test_data.dropna(axis=1)

str_cols = test_data.columns[test_data.columns.str.contains('Sex')]
clfs = {c:LabelEncoder() for c in str_cols}
for col, clf in clfs.items():
    test_data[col] = clfs[col].fit_transform(test_data[col])

test_X = test_data[features]
test_predictions = titanic_model.predict(test_X)
test_predictions = titanic_model.predict(test_X)

int_test_predictions = test_predictions.astype(int)


my_result = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': int_test_predictions})
my_result.to_csv('submission.csv', index=False)

