#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u"config InlineBackend.figure_format = 'retina'")

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
import seaborn as sns

import itertools
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# In[ ]:


def easy_drop(tdf: pd.DataFrame, *columns):
    if len(columns) == 1 and (isinstance(columns[0], (list, tuple))):
        columns = columns[0]
    for col in columns:
        if col in tdf.columns:
            tdf.drop(col, axis=1, inplace=True)
    return tdf
def as_categorical(tdf, *columns):
    if len(columns) == 1 and (isinstance(columns[0], (list, tuple))):
        columns = columns[0]
    for col in columns:
        if col in tdf.columns:
            tdf[col] = tdf[col].astype('category')
def load(filepath):
    tdf = pd.read_csv(filepath)
    tdf.columns = tdf.columns.str.lower()
    return tdf.set_index('passengerid', verify_integrity=True)


# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


raw = load('../input/train.csv')
test_df = load('../input/test.csv')
full_df = pd.concat([raw, test_df])


# In[ ]:


ldf = raw.sort_values(['survived', 'age']).reset_index(drop=1)
xline = ldf[ldf.survived == 0].survived.count()

ldf.age.plot()
plt.axvline(xline, c='g');


# In[ ]:


raw.head()


# In[ ]:


sns.distplot(full_df.age.fillna(0));


# In[ ]:


full_df['title'] = full_df.name.str.extract(r'^[^,]+, ([\S]*) .*$', expand=False).astype('category')
full_df.groupby('title').survived.mean().to_frame()


# # Base model

# In[ ]:


allways_yes = raw.survived.sum() / raw.survived.count()
print('Base accuracy: {:.2f}'.format(max(allways_yes, 1-allways_yes)))


# # Feature engineering

# In[ ]:


full_df['lastname'] = full_df.name.str.extract("^([a-zA-Z \-']+),", expand=False)
full_df['relatives_survived'] = full_df.groupby('lastname', as_index=False)    .survived.transform('sum').subtract(full_df.survived, axis=0)


# In[ ]:


def generate_features(tdf: pd.DataFrame):
    features = []
    features.append('pclass')
    features.append('sibsp')
    features.append('parch')
    features.append('age')
    features.append('fare')
#     features.append('survived')
    
    tdf = tdf.copy()
    
    tdf['age'] = tdf.age.fillna(tdf.age.median())
    tdf['fare'] = tdf.fare.fillna(tdf.age.median())
    
    tdf['third_class'] = tdf.pclass == 3
    features.append('third_class')

    tdf['male'] = (tdf.sex == 'male')
    features.append('male')

    tdf['cabin_letter_cat'] = tdf.cabin.str[0].astype('category').cat.codes
    features.append('cabin_letter_cat')

    tdf['embarked_code'] = tdf.embarked.astype('category').cat.codes
    features.append('embarked_code')
    
    tdf['rounded_fare'] = tdf.fare.fillna(tdf.fare.mean()).round(decimals=-1).astype(np.int32)
    features.append('rounded_fare')
    
    tdf['fare_log'] = np.log(tdf.fare.fillna(tdf.fare.median())+0.1)
    features.append('fare_log')
    
    tdf['age_f'] = tdf.age.fillna(tdf.age.mean())
    features.append('age_f')
    
    tdf['age_cat'] = pd.cut(tdf.age, np.arange(0, raw.age.max()+1, 5)).cat.codes
    features.append('age_cat')
    
    tdf['words_in_name'] = tdf.name.str.split().apply(lambda x: len(x))
    features.append('words_in_name')

    tdf['lastname'] = tdf.name.str.extract("^([a-zA-Z \-']+),", expand=False)
    
    tdf['relatives_survived'] = tdf.groupby('lastname', as_index=False)    .survived.transform('sum').subtract(full_df.survived.fillna(0), axis=0)
    features.append('relatives_survived')
    
    tdf['title_cat'] = tdf.name.str.extract(r'^[^,]+, ([\S]*) .*$', expand=False).astype('category').cat.codes
    features.append('title_cat')
    
    return tdf[features].copy()


# # Training

# In[ ]:


test_df = load('../input/test.csv')
full_df = pd.concat([raw, test_df])

X = generate_features(full_df).loc[raw.index, :]
y = raw.survived
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

model = RandomForestClassifier(max_depth=10, 
                               n_estimators=50, 
                               max_features='auto', 
                               criterion='gini',
                               random_state=42, n_jobs=-1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42, stratify=y)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Mean accuracy: {:.4f}'.format(model.score(X_test, y_test)))
print(classification_report(y_test, y_pred, target_names=['died', 'survived']))
plot_confusion_matrix(confusion_matrix(y_test, y_pred), ['died', 'survived'])

model.fit(X_test, y_test);


# In[ ]:


max_len = X.columns.str.len().max()
for imp, f in sorted(zip(model.feature_importances_, X.columns)): print('{:{len}}: {:.3f}'.format(f, imp, len=max_len))
    
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]]);

