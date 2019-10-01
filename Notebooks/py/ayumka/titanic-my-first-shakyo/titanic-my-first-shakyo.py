#!/usr/bin/env python
# coding: utf-8

# # Thank you ldfreeman3!
# Hi, I'm Ayumu from Japan! I'm interested in python & machine learning. Thanks to [ldfreeman3's beginner-friendly kernel](https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy/notebook), I could walk into kaggle<3 
# 
# I could reach submitting CSV, but almost all parts I still haven't understood:(  So I will learn the contents more deeply and add comment on my kernel little by little, day by day ⌒°(๑°◡°๑)°⌒ I welcome and appreciate any comments!
# 
# ldfreeman3さん、ありがとうございます(*≧∀≦) あなたのおかげでTitanicの生存者を予測して提出するところまで漕ぎ着けることができました! しかし写経なので内容がまだまったく頭に入っていないので、これから少しずつコードを解釈していこうと思いますＹ(๑°口°๑) どんなコメントも歓迎です^^/
# 
# # import sys,  __version__, warnings...?
# In[1] の解説をします。コードが見たい方は下へとスクロールしてください。
# `**import sys**`
# 
# `sys` とは、インタプリタや実行環境に関連した変数や関数がまとめられたPython の標準ライブラリです。`sys.path` でモジュール検索パスを確認可能。以下のcodeセルでは`sys.version` を使ってPythonのバージョンを表示させています。
# 
# `**print("pandas version: {}".format(pd.__version__))**`
# 
# __version__属性でバージョン番号が取得できます。`"文字列".format()` と書くと`.format()` 内の情報を`{}`があるところに表示してくれます。
# 
# 
# 
# ```python
# import warnings
# warnings.filterwarnings('ignore')
# ```
# 
# UserWarningなど、表示しなくても問題ない警告はこれを使うと表示されなくなります。スクロースが必要なほどずらずら警告が出てきたセルにこれを入れるときれいさっぱり表示されなくなって重宝します。
# 
# さて、これらの前提知識を仕入れて次のcodeセルを読んでみてください。

# In[ ]:


import sys
print("Python version: {}".format(sys.version))
import pandas as pd
print("pandas version: {}".format(pd.__version__))
import matplotlib
print("matplotlib version: {}".format(matplotlib.__version__))

import numpy as np
print("Numpy version: {}".format(np.__version__))
import scipy as sp
print("SciPy version: {}".format(sp.__version__))
import IPython
from IPython import display
print("IPython version: {}".format(IPython.__version__))

import sklearn
print("scikit-learn version: {}".format(sklearn.__version__))
import random
import time

import warnings
warnings.filterwarnings('ignore')
print('-'*25)

import os
print(os.listdir('../input'))


# # sklearn、xgboost、matplotlibなどをインポート
# In[2]の解説をしていきます。
# ## sklearnからimportするもの
# ```python
# from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
# ```
# ここでインポートするモジュールは8個です。
# ちなみにsklearnはPythonの機械学習ライブラリです。
# 1.	sklearn.svm
# Support Vector Machine (SVM) アルゴリズムを扱うモジュール。
# SVM: 教師あり学習を用いるパターン認識モデル。
# もっと見る: [sklearn.svm](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm)
# 2.	sklearn.tree
# 分類と回帰用の決定木ベースモデルを扱うモジュール。
# もっと見る: [sklearn.tree](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree)
# 3.	sklearn.linear_model
# 一般化線形モデル (generalized linear model) を実行するモジュール。
# もっと見る: [sklearn.linear_model](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model)
# 4.	sklearn.neighbors
# k近傍法 (k-nearest neighbor algorithm) を実行するモジュール。k近傍法とは特徴空間における最も近い訓練例に基づいた分類の手法であり、パターン認識でよく使われる。最近傍探索問題の一つ。
# もっと見る: [sklearn.neighbors](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors)
# 5.	sklearn.naive_bayes
# 単純ベイズ (Naive Bayes) アルゴリズムを実行するモジュール。
# もっと見る: [sklearn.naive_bayes](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.naive_bayes)
# 6.	sklearn.ensemble
# 分類、回帰、異常検知 (anomaly detection) 用のアンサンブルベースメソッドを実行するモジュール。
# もっと見る: [sklearn.ensemble](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble)
# 7.	sklearn.discriminant_analysis
# 線形判別分析 (Linear Discriminant Analysis, LDA) と2次判別 (Quadratic Discriminant Analysis）を実行するモジュール。
# もっと見る: [sklearn.discriminant_analysis](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.discriminant_analysis)
# 8.	sklearn.gaussian_process
# 回帰と分類に基づいたガウス過程を実行するモジュール。
# もっと見る: [sklearn.gaussian_process](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.gaussian_process)
# 
# ## xgboostからXGBClassifierをインポート
# XGBoostは、高速な処理、外れ値や欠損値に強い点が持ち味らしいのですが、残念ながらよくわかりませんでしたf^^; 時間が溶けるので飛ばします。上手に説明がしてあるサイトなどあれば教えてくださいm(_)m 
# XGBoostは、最も有力で・競争力のある機械学習である速さとパフォーマンスのために設計された勾配ブースト決定木の実装です。モデルの訓練に使います。
# XGBoost参考サイト: [Machine Learning Mastery](https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/)
# ## sklearn.preprocessingからOneHotEncoderとLabelEncoder をインポート
# 
#  sklearn.preprocessingパッケージは、生データを前処理するために使います。
#  - OneHotEncoder: カテゴリの整数特性をOne Hot表現 (1つだけHigh(1)であり、他はLow(0)であるようなビット列) を使ってエンコードします。
#  - LabelEncoder: 0 と n_classes-1 間の値でラベルをエンコードします。
#  
#  ## もう一度sklearnからimportするもの
#  - feature_selection: 特徴選択アルゴリズムを実装するモジュール。
#  - model_selection: model_selectionそのものについて説明しているものなし。
#  - metrics: 作成したモデルの評価を行うモジュール
#  
#  # matplotlibとは
# 
# 

# In[ ]:


from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

get_ipython().magic(u'matplotlib inline')
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8


# In[ ]:


# import data from file
data_raw = pd.read_csv('../input/train.csv')

data_val = pd.read_csv('../input/test.csv')
data1 = data_raw.copy(deep = True)
data_cleaner = [data1, data_val]
print(data_raw.info())
data_raw.sample(10)


# In[ ]:


print('Train columns with null values:\n', data1.isnull().sum())
print("-"*10)

print('Test/Validation columns with null values:\n', data_val.isnull().sum())
print("-"*10)

data_raw.describe(include = 'all')


# In[ ]:


for dataset in data_cleaner:
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
    
drop_column = ['PassengerId','Cabin', 'Ticket']
data1.drop(drop_column, axis=1, inplace = True)

print(data1.isnull().sum())
print("-"*10)
print(data_val.isnull().sum())


# In[ ]:


for dataset in data_cleaner:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
    dataset['IsAlone'] = 1
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0
    
    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)
    
stat_min = 10
title_names = (data1['Title'].value_counts() < stat_min)

data1['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
print(data1['Title'].value_counts())
print("-"*10)

data1.info()
data_val.info()
data1.sample(10)


# In[ ]:


label = LabelEncoder()
for dataset in data_cleaner:
    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
    dataset['Title_Code'] = label.fit_transform(dataset['Title'])
    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])
    
Target = ['Survived']

data1_x = ['Sex', 'Pclass', 'Embarked', 'Title', 'SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone']
data1_x_calc = ['Sex_Code', 'Pclass', 'Embarked_Code', 'Title_Code', 'SibSp', 'Parch', 'Age', 'Fare']
data1_xy = Target + data1_x
print('Original X Y: ', data1_xy, '\n')

data1_x_bin = ['Sex_Code', 'Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
data1_xy_bin = Target + data1_x_bin
print('Bin X Y: ', data1_xy_bin, '\n')

data1_dummy = pd.get_dummies(data1[data1_x])
data1_x_dummy = data1_dummy.columns.tolist()
data1_xy_dummy = Target + data1_x_dummy
print('Dummy X Y: ', data1_xy_dummy, '\n')

data1_dummy.head()


# In[ ]:


# 3.24 Da-Double Check Cleaned Data
print('Train columns with null values: \n', data1.isnull().sum())
print("-"*10)
print(data1.info())
print("-"*10)

print('Test/Validation columns with null values: \n', data_val.isnull().sum())
print("-"*10)
print(data_val.info())
print("-"*10)

data_raw.describe(include = 'all')


# In[ ]:


# 3.25 Split Training and Testing Data
train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(data1[data1_x_calc], data1[Target], random_state = 0)
train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(data1[data1_x_bin], data1[Target] , random_state = 0)
train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(data1_dummy[data1_x_dummy], data1[Target], random_state = 0)


print("Data1 Shape: {}".format(data1.shape))
print("Train1 Shape: {}".format(train1_x.shape))
print("Test1 Shape: {}".format(test1_x.shape))

train1_x_bin.head()


# # Step 4: Perform Exploratory Analysis with Statistics

# In[ ]:


for x in data1_x:
    if data1[x].dtype != 'float64':
        print('Survival Correction by:', x)
        print(data1[[x, Target[0]]].groupby(x, as_index = False).mean())
        print('-' * 10, '\n')
        
print(pd.crosstab(data1['Title'], data1[Target[0]]))


# In[ ]:


#graph distribution of quantitative data
plt.figure(figsize=[16, 12])

plt.subplot(231)
plt.boxplot(x=data1['Fare'], showmeans = True, meanline = True)
plt.title('Fare Boxplot')
plt.ylabel('Fare ($)')

plt.subplot(232)
plt.boxplot(data1['Age'], showmeans = True, meanline = True)
plt.title('Age Boxplot')
plt.ylabel('Age (Years)')

plt.subplot(233)
plt.boxplot(x=data1['FamilySize'], showmeans = True, meanline = True)
plt.title('Family Size Boxplot')
plt.ylabel('Family Size (#)')

plt.subplot(234)
plt.hist(x = [data1[data1['Survived']==1]['Fare'], data1[data1['Survived']==0]['Fare']], stacked=True, color = ['g', 'r'], label = ['Survived', 'Dead'])
plt.title('Fare Histgram by Survival')
plt.xlabel('Fare ($)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(235)
plt.hist(x = [data1[data1['Survived']==1]['Age'], data1[data1['Survived']==0]['Age']], stacked=True, color = ['g', 'r'], label = ['Survived', 'Dead'])
plt.title('Age Histgram by Survival')
plt.xlabel('Age (Years)')
plt.ylabel('# of Passengers')
plt.legend()

plt.subplot(236)
plt.hist(x = [data1[data1['Survived']==1]['FamilySize'], data1[data1['Survived']==0]['FamilySize']], stacked=True, color = ['g', 'r'], label = ['Survived', 'Dead'])
plt.title('Family Size Histgram by Survival')
plt.xlabel('Family Size (#)')
plt.ylabel('# of Passengers')
plt.legend()


# In[ ]:


#graph individual features by survival
fig, saxis = plt.subplots(2, 3, figsize=(16, 12))

sns.barplot(x = 'Embarked', y = 'Survived', data = data1, ax = saxis[0, 0])
sns.barplot(x = 'Pclass', y = 'Survived', order = [1, 2, 3], data = data1, ax = saxis[0, 1])
sns.barplot(x = 'IsAlone', y = 'Survived', order = [1, 0], data = data1, ax = saxis[0, 2])

sns.pointplot(x = 'FareBin', y = 'Survived', data = data1, ax = saxis[1, 0])
sns.pointplot(x = 'AgeBin', y = 'Survived', data = data1, ax = saxis[1, 1])
sns.pointplot(x = 'FamilySize', y = 'Survived', data = data1, ax = saxis[1, 2])


# In[ ]:


#we know class mattered in survival, now let's compare class and a 2nd feature
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(14,12))

sns.boxplot(x = 'Pclass', y = 'Fare', hue = 'Survived', data = data1, ax = axis1)
axis1.set_title('Pclass vs Fare Survival Comparison')

sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', data = data1, split = True, ax = axis2)
axis2.set_title('Pclass vs Age Survival Comparison')

sns.boxplot(x = 'Pclass', y = 'FamilySize', hue = 'Survived', data = data1, ax = axis3)
axis3.set_title('Pclass vs Family Size Survival Comparison')


# In[ ]:


#we know sex mattered in survival, now let's compare sex and a 2nd feature
fig, qaxis = plt.subplots(1, 3, figsize=(14, 12))

sns.barplot(x = 'Sex', y = 'Survived', hue = 'Embarked', data = data1, ax = qaxis[0])
axis1.set_title('Sex vs Embarked Survival Comparison')

sns.barplot(x = 'Sex', y = 'Survived', hue = 'Pclass', data = data1, ax = qaxis[1])
axis1.set_title('Sex vs Pclass Survival Comparison')

sns.barplot(x = 'Sex', y = 'Survived', hue = 'IsAlone', data = data1, ax = qaxis[2])
axis1.set_title('Sex vs IsAlone Survival Comparison')


# In[ ]:


#more side-by-side comparisons
fig, (maxis1, maxis2) = plt.subplots(1, 2, figsize=(14, 12))

sns.pointplot(x = "FamilySize", y = "Survived", hue = "Sex", data = data1,
              palette= {"male": "blue", "female": "pink"},
              markers=["*", "o"], linestyles=["-", "--"], ax = maxis1)

sns.pointplot(x = "Pclass", y = "Survived", hue = "Sex", data = data1,
              palette= {"male": "blue", "female": "pink"},
              markers=["*", "o"], linestyles=["-", "--"], ax = maxis2)


# In[ ]:


#how does embark port factor with class, sex, and survival compare
e = sns.FacetGrid(data1, col = 'Embarked')
e.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', ci=95.0, palette = 'deep',
         order = [1,2,3], hue_order=["female", "male"])
e.add_legend()


# In[ ]:


#plot distributions of age of passengers who survived or did not survive
a = sns.FacetGrid( data1, hue = 'Survived', aspect = 4 )
a.map(sns.kdeplot, 'Age', shade = True )
a.set(xlim=(0, data1['Age'].max()))
a.add_legend()


# In[ ]:


#histogram comparison of sex, class, and age by survival
h = sns.FacetGrid(data1, row = 'Sex', col = 'Pclass', hue = 'Survived')
h.map(plt.hist, 'Age', alpha = .75)
h.add_legend()


# In[ ]:


#pair plots of entire dataset
pp = sns.pairplot(data1, hue = 'Survived', palette = 'deep', size = 1.2, diag_kind = 'kde', diag_kws=dict(shade=True), plot_kws=dict(s=10))
pp.set(xticklabels=[])


# In[ ]:


#correlation heatmap of dataset
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(),
        cmap = colormap,
        square = True,
        cbar_kws={'shrink' :.9},
        ax = ax,
        annot=True,
        linewidths= 0.1, vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12}
        )
    
    plt.title('Pearson Correlation of Features', y = 1.05, size = 15)
    
correlation_heatmap(data1)


# # Step 5: Model Data

# In[ ]:


#Machine Learning Algorithm (MLA) Selection and Initialization
MLA = [
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),
    
    gaussian_process.GaussianProcessClassifier(),
    
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    neighbors.KNeighborsClassifier(),
    
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),
    
    XGBClassifier()
]

cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0)

MLA_columns = ['MLA Name', 'MLA Parameters', 'MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD', 'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

MLA_predict = data1[Target]

row_index = 0
for alg in MLA:
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    
    cv_results = model_selection.cross_validate(alg, data1[data1_x_bin], data1[Target].values.ravel(), cv = cv_split)
    
    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()
    
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3
    
    alg.fit(data1[data1_x_bin], data1[Target].values.ravel())
    MLA_predict[MLA_name] = alg.predict(data1[data1_x_bin])
    
    row_index += 1
    
    
MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
MLA_compare
    


# In[ ]:





# In[ ]:


sns.barplot(x='MLA Test Accuracy Mean', y = 'MLA Name', data = MLA_compare, color = 'm')

plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')


# # 5.1 Evaluate Model Performance

# In[ ]:


#IMPORTANT: This is a handmade model for learning purposes only.
for index, row in data1.iterrows():
    if random.random() > .5:
        data1.set_value(index, 'Random_Predict', 1)
    else:
        data1.set_value(index, 'Random_Predict', 0)
        
data1['Random_Score'] = 0
data1.loc[(data1['Survived'] == data1['Random_Predict']), 'Random_Score'] = 1
print('Coin Flip Model Accuracy: {:.2f}%'.format(data1['Random_Score'].mean()*100))

print('Coin Flip Model Accuracy w/SciKit: {:.2f}%'.format(metrics.accuracy_score(data1['Survived'], data1['Random_Predict'])*100))


# In[ ]:


#group by or pivot table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html
pivot_female = data1[data1.Sex == 'female'].groupby(['Sex', 'Pclass', 'Embarked', 'FareBin'])['Survived'].mean()
print('Survival Decision Tree w/Female Node: \n', pivot_female)

pivot_male = data1[data1.Sex == 'male'].groupby(['Sex', 'Title'])['Survived'].mean()
print('\n\nSurvival Decision Tree w/Male Node: \n', pivot_male)


# In[ ]:


#handmade data model using brain power (and Microsoft Excel Pivot Tables for quick calculations)
def mytree(df):
    Model = pd.DataFrame(data = {'Predict':[]})
    male_title = ['Master']
    
    for index, row in df.iterrows():
        Model.loc[index, 'Predict'] = 0
        
        if (df.loc[index, 'Sex'] == 'female'):
            Model.loc[index, 'Predict'] = 1
            
        if ((df.loc[index, 'Sex'] == 'female') &
            (df.loc[index, 'Pclass'] == 3) &
            (df.loc[index, 'Embarked'] == 'S') &
            (df.loc[index, 'Fare'] > 8)
           ):
                Model.loc[index, 'Predict'] = 0
                
        if ((df.loc[index, 'Sex'] == 'male') &
            (df.loc[index, 'Title'] in male_title)
           ):
           Model.loc[index, 'Predict'] = 1
    return Model

Tree_Predict = mytree(data1)
print('Decision Tree Model Accuracy/Prediction Score: {:.2}%\n'.format(metrics.accuracy_score(data1['Survived'], Tree_Predict)*100))

print(metrics.classification_report(data1['Survived'], Tree_Predict))


# In[ ]:


#Plot Accuracy Summary
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize = False,
                          title = 'Confusion matrix',
                          cmap = plt.cm.Blues):
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        
    print(cm)
    
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment = "center",
                 color = "white" if cm[i, j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
cnf_matrix = metrics.confusion_matrix(data1['Survived'], Tree_Predict)
np.set_printoptions(precision = 2)

class_names = ['Dead', 'Survived']
plt.figure()
plot_confusion_matrix(cnf_matrix, classes = class_names, 
                      title = 'Confusion matrix, without normalization')

plt.figure()
plot_confusion_matrix(cnf_matrix, classes = class_names, normalize = True,
                      title = 'Normalized confusion matrix')


# In[ ]:


#base model
dtree = tree.DecisionTreeClassifier(random_state = 0)
base_results = model_selection.cross_validate(dtree, data1[data1_x_bin], data1[Target], cv = cv_split)
dtree.fit(data1[data1_x_bin], data1[Target])

print('BEFORE DT Parameters: ', dtree.get_params())
print('BEFORE DT Training w/bin score mean: {:.2f}'.format(base_results['train_score'].mean()*100))
print('BEFORE DT Test w/bin score mean: {:.2f}'.format(base_results['test_score'].mean()*100))
print('BEFORE DT Test w/bin score 3*std: +/- {:.2f}'.format(base_results['test_score'].std()*100*3))

print('-'*10)

param_grid = {'criterion': ['gini', 'entropy'],
              'max_depth': [2, 4, 6, 8, 10, None],
              'random_state': [0]
             }

tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid = param_grid, scoring = 'roc_auc', cv = cv_split)
tune_model.fit(data1[data1_x_bin], data1[Target])

print('AFTER DT Parameters: ', tune_model.best_params_)
print('AFTER DT Training w/bin score mean: {:.2f}'.format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100))
print('AFTER DT Test w/bin score mean: {:.2f}'.format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
print('AFTER DT Test w/bin score 3*std: +/- {:.2f}'.format(tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))
print('-'*10)


# # 5.13 Tune Model with Feature Selection

# In[ ]:


print('BEFORE DT RFE Training Shape Old: ', data1[data1_x_bin].shape)
print('BEFORE DT RFE Training Columns Old: ', data1[data1_x_bin].columns.values)

print('BEFORE DT RFE Training w/bin score mean: {:.2f}'.format(base_results['train_score'].mean()*100))
print('BEFORE DT RFE Test w/bin score mean: {:.2f}'.format(base_results['test_score'].mean()*100))
print('BEFORE DT RFE Training w/bin score 3*std: +/- {:.2f}'.format(base_results['test_score'].std()*100*3))
print('-' * 10)

dtree_rfe = feature_selection.RFECV(dtree, step = 1, scoring = 'accuracy', cv = cv_split)
dtree_rfe.fit(data1[data1_x_bin], data1[Target].values.ravel())

X_rfe = data1[data1_x_bin].columns.values[dtree_rfe.get_support()]
rfe_results = model_selection.cross_validate(dtree, data1[X_rfe], data1[Target].values.ravel(), cv = cv_split)

print('AFTER DT RFE Training Shape New: ', data1[X_rfe].shape)
print('AFTER DT RFE Training Columns New: ', X_rfe)

print('AFTER DT RFE Training w/bin score mean: {:.2f}'.format(rfe_results['train_score'].mean()*100))
print('AFTER DT RFE Test w/bin score mean: {:.2f}'.format(rfe_results['test_score'].mean()*100))
print('AFTER DT RFE Training w/bin score 3*std: +/- {:.2f}'.format(rfe_results['test_score'].std()*100*3))
print('-'*10)

rfe_tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid = param_grid, scoring = 'roc_auc', cv = cv_split)
rfe_tune_model.fit(data1[X_rfe], data1[Target].values.ravel())

print('AFTER DT RFE Tuned Parameters: ', rfe_tune_model.best_params_)
print('AFTER DT RFE Tuned Training w/bin score mean: {:.2f}'.format(rfe_tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100))
print('AFTER DT RFE Tuned Test w/bin score mean: {:.2f}'.format(rfe_tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
print('AFTER DT RFE TunedTraining w/bin score 3*std: +/- {:.2f}'.format(rfe_tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))
print('-'*10)


# In[ ]:


#Graph MLA version of Decision Tree: http://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html
import graphviz
dot_data = tree.export_graphviz(dtree, out_file = None,
                                feature_names = data1_x_bin, class_names = True,
                                filled = True, rounded = True)
graph = graphviz.Source(dot_data)
graph


# # Step 6: Validate and Implement

# In[ ]:


correlation_heatmap(MLA_predict)


# In[ ]:


#why choose one model, when you can pick them all with voting classifier
vote_est = [
    ('ada', ensemble.AdaBoostClassifier()),
    ('bc', ensemble.BaggingClassifier()),
    ('etc', ensemble.ExtraTreesClassifier()),
    ('gbc', ensemble.GradientBoostingClassifier()),
    ('rfc', ensemble.RandomForestClassifier()),
    
    ('gpc', gaussian_process.GaussianProcessClassifier()),
    ('lr', linear_model.LogisticRegressionCV()),
    
    ('bnb', naive_bayes.BernoulliNB()),
    ('gnb', naive_bayes.GaussianNB()),
    
    ('knn', neighbors.KNeighborsClassifier()),
    ('svc', svm.SVC(probability = True)),
    
    ('xgb', XGBClassifier())
]

vote_hard = ensemble.VotingClassifier(estimators = vote_est, voting = 'hard')
vote_hard_cv = model_selection.cross_validate(vote_hard, data1[data1_x_bin], data1[Target].values.ravel(), cv = cv_split)
vote_hard.fit(data1[data1_x_bin], data1[Target].values.ravel())

print("Hard Voting Training w/bin score mean: {:.2f}".format(vote_hard_cv['train_score'].mean()*100))
print("Hard Voting Test w/bin score mean: {:.2f}".format(vote_hard_cv['test_score'].mean()*100))
print("Hard Voting Test w/bin score 3*std: +/- {:.2f}".format(vote_hard_cv['test_score'].std()*100*3))
print('-'*10)

vote_soft = ensemble.VotingClassifier(estimators = vote_est, voting = 'soft')
vote_soft_cv = model_selection.cross_validate(vote_soft, data1[data1_x_bin], data1[Target].values.ravel(), cv = cv_split)
vote_soft.fit(data1[data1_x_bin], data1[Target].values.ravel())

print("Soft Voting Training w/bin score mean: {:.2f}".format(vote_soft_cv['train_score'].mean()*100))
print("Soft Voting Test w/bin score mean: {:.2f}".format(vote_soft_cv['test_score'].mean()*100))
print("Soft Voting Test w/bin score 3*std: +/- {:.2f}".format(vote_soft_cv['test_score'].std()*100*3))
print('-'*10)


# In[ ]:


#WARNING: Running is very computational intensive and time expensive.
import warnings
warnings.filterwarnings('ignore')

grid_n_estimator = [10, 50, 100, 300]
grid_ratio = [.1, .25, .5, .75, 1.0]
grid_learn = [.01, .03, .05, .1, .25]
grid_max_depth = [2, 4, 6, 8, 10, None]
grid_min_samples = [5, 10, .03, .05, .10]
grid_criterion = ['gini', 'entropy']
grid_bool = [True, False]
grid_seed = [0]

grid_param = [
                [{
                    'n_estimators': grid_n_estimator,
                    'learning_rate': grid_learn,
                    'random_state': grid_seed
                }],
    
                [{
                    'n_estimators': grid_n_estimator,
                    'max_samples': grid_ratio,
                    'random_state': grid_seed
                }],

                [{
                    'n_estimators': grid_n_estimator,
                    'criterion': grid_criterion,
                    'max_depth': grid_max_depth,
                    'random_state': grid_seed
                }],

                [{
                    'learning_rate': [.05],
                    'n_estimators': [300],
                    'max_depth': grid_max_depth,
                    'random_state': grid_seed
                }],

                [{
                    'n_estimators': grid_n_estimator,
                    'criterion': grid_criterion,
                    'max_depth': grid_max_depth,
                    'oob_score': [False],
                    'random_state': grid_seed
                }],
                
                [{
                    'max_iter_predict': grid_n_estimator,
                    'random_state': grid_seed
                }],
    
                [{
                    'fit_intercept': grid_bool,
                    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                    'random_state': grid_seed
                }],
    
                [{
                    'alpha': grid_ratio,
                }],
    
                [{}],
    
                [{
                    'n_neighbors': [1,2,3,4,5,6,7],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                }],
    
                [{
                    'C': [1,2,3,4,5],
                    'gamma': grid_ratio,
                    'decision_function_shape': ['ovo', 'ovr'],
                    'probability': [True],
                    'random_state': grid_seed
                }],
    
                [{
                    'learning_rate': grid_learn,
                    'max_depth': [1,2,4,6,8,10],
                    'n_estimators': grid_n_estimator,
                    'seed': grid_seed
                }]
]


start_total = time.perf_counter()
for clf, param in zip (vote_est, grid_param):
    start = time.perf_counter()
    best_search = model_selection.GridSearchCV(estimator = clf[1], param_grid = param, cv = cv_split, scoring = 'roc_auc')
    best_search.fit(data1[data1_x_bin], data1[Target].values.ravel())
    run = time.perf_counter() - start
    
    best_param = best_search.best_params_
    print('The best parameter for {} is {}  with a runtime of {:.2f} seconds'.format(clf[1].__class__.__name__, best_param, run))
    clf[1].set_params(**best_param)
    
run_total = time.perf_counter() - start_total
print('Total optimization time was {:.2f} minutes.'.format(run_total/60))

print('-' *10)


# In[ ]:


#Hard Vote or majority rules w/Tuned Hyperparameters
grid_hard = ensemble.VotingClassifier(estimators = vote_est, voting = 'hard')
grid_hard_cv = model_selection.cross_validate(grid_hard, data1[data1_x_bin], data1[Target].values.ravel(), cv = cv_split)
grid_hard.fit(data1[data1_x_bin], data1[Target].values.ravel())

print("Hard Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}".format(grid_hard_cv['train_score'].mean()*100))
print("Hard Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}".format(grid_hard_cv['test_score'].mean()*100))
print("Hard Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}".format(grid_hard_cv['test_score'].std()*100*3))
print('-'*10)

grid_soft = ensemble.VotingClassifier(estimators = vote_est, voting = 'soft')
grid_soft_cv = model_selection.cross_validate(grid_soft, data1[data1_x_bin], data1[Target].values.ravel(), cv = cv_split)
grid_soft.fit(data1[data1_x_bin], data1[Target].values.ravel())

print("Soft Voting w/Tuned Hyperparameters Training w/bin score mean: {:.2f}".format(grid_soft_cv['train_score'].mean()*100))
print("Soft Voting w/Tuned Hyperparameters Test w/bin score mean: {:.2f}".format(grid_soft_cv['test_score'].mean()*100))
print("Soft Voting w/Tuned Hyperparameters Test w/bin score 3*std: +/- {:.2f}".format(grid_soft_cv['test_score'].std()*100*3))
print('-'*10)


# In[ ]:


#prepare data for modeling
print(data_val.info())
print("-"*10)

data_val['Survived'] = mytree(data_val).astype(int)
data_val['Survived'] = grid_hard.predict(data_val[data1_x_bin])

submit = data_val[['PassengerId', 'Survived']]
submit.to_csv("../working/submit.csv", index=False)

print('Validation Data Distribution: \n', data_val['Survived'].value_counts(normalize = True))
submit.sample(10)


# In[ ]:




