#!/usr/bin/env python
# coding: utf-8

# 
# 
# ```python
# #整个的流程如下
# '''
# 1.读取训练集和测试集，利用concat进行合并，方便一起进行数据处理
# 
# 2.进行数据的第一次处理，删除ID号和目标值，此时把训练集中的目标值先取出来，留到后面训练模型使用
# 
# 3.对数据的缺失值进行了解，这个案例中，有三个缺失值，分别是Cabin,Age和Embark,Fare。
#    对于Cabin而言，属于类别型，而且缺失值较多，一开始考虑丢弃，但是想着把有无值作为一个特征进行看待，发现有Cabin值的存活率高，或许本身也有不恰当的地方，但是先这么处理了
#    对于Embark而言，属于类别型，缺失值很少，考虑用众数进行填充；
#    对于Fare而言，缺失值只有一个，通过常识和数据分析判断，Fare的值与船舱等级关系较大，缺失Fare的这一行训练数据中，PClass是完整的，利用所在PClass的均值进行填充
#    对于Age而言，缺失量一般，而且Age对目标值的影响较大，应该要重点研究，本例中利用均值进行填充了，可以研究Age与其他特征的关系，比如与姓名中称呼的关系等，利用算法进行填充
# 
# 4.至此，整个数据已经没有缺失值，开始考虑下一步操作。
#   第一：已有特征的合并，比如直系家属和旁系家属可以合并为家庭规模
#   第二：删除不需要的特征，包括合并前的两个特征，以及Ticket,Name等与目标值无关的特征。在这里，需要说明一下，名字特征中可以提取出来Title属性，貌似对目标值有影响，可以考虑
#   第三：连续型数值的离散化，比如年龄，其实可以分为小孩，老人，等等
#   第三：整个数据集进行one-hot操作，类别型的进行0，1编码。在其他的案例中有看到不进行此操作的，比如Pclass,就直接保留原有值，而不进行one-hot。
#         需要强调的是，本身Pclass的数值只是作为类别区分，不是真正的数值，如果在线性计算模型中，这些数据就会影响实际的判断，因此，按照实际的情况进行one-hot编码最好
#     
# 5.把处理完成的数据集分成原有的训练集和测试集
# 6.引入机器学习算法模型，对模型进行训练并且预测测试集。
# 
# 
# 还需要改进的地方：
# 1.对于年龄的填充需要采用更合理的方法
# 2.对于年龄的离散化可以进行考虑
# 3.模型的交叉验证进行调参是关键，需要进行下一步工作
# 4.模型融合需要进行考虑
# 5.进一步熟悉数据分析的方式和方法
#    
# 
# '''
# # Load in our libraries
# import pandas as pd
# import numpy as np
# import re
# import sklearn
# import xgboost as xgb
# import seaborn as sns
# import matplotlib.pyplot as plt
# %matplotlib inline
# 
# 
# import warnings
# warnings.filterwarnings('ignore')
# ```
# 
# 
# ```python
# data_train=pd.read_csv("D:\ML_Data\Tianic\Train.csv")
# ```
# 
# 
# ```python
# data_test=pd.read_csv("D:\ML_Data\Tianic\Test.csv")
# ```
# 
# 
# ```python
# data_all=pd.concat([data_train,data_test],ignore_index=True)
# y=data_train['Survived']
# ```
# 
# 
# ```python
# data_all.info()
# ```
# 
#     <class 'pandas.core.frame.DataFrame'>
#     RangeIndex: 1309 entries, 0 to 1308
#     Data columns (total 12 columns):
#     Age            1046 non-null float64
#     Cabin          295 non-null object
#     Embarked       1307 non-null object
#     Fare           1308 non-null float64
#     Name           1309 non-null object
#     Parch          1309 non-null int64
#     PassengerId    1309 non-null int64
#     Pclass         1309 non-null int64
#     Sex            1309 non-null object
#     SibSp          1309 non-null int64
#     Survived       891 non-null float64
#     Ticket         1309 non-null object
#     dtypes: float64(3), int64(4), object(5)
#     memory usage: 122.8+ KB
#     
# 
# 
# ```python
# data_all.isnull().sum().sort_values(ascending=False)
# ```
# 
# 
# 
# 
#     Cabin          1014
#     Survived        418
#     Age             263
#     Embarked          2
#     Fare              1
#     Ticket            0
#     SibSp             0
#     Sex               0
#     Pclass            0
#     PassengerId       0
#     Parch             0
#     Name              0
#     dtype: int64
# 
# 
# 
# 
# ```python
# data_train.loc[(data_train.Cabin.notnull()),'Cabin']="Yes"
# ```
# 
# 
# ```python
# data_train.loc[(data_train.Cabin.isnull()),'Cabin']="NO"
# ```
# 
# 
# ```python
# data_train.groupby(['Cabin'])['Survived'].mean().plot(kind='bar')
# ```
# 
# 
# 
# 
#     <matplotlib.axes._subplots.AxesSubplot at 0xc7d8d30>
# 
# 
# 
# 
# ![png](output_8_1.png)
# 
# 
# 
# ```python
# data_all.loc[(data_all.Cabin.notnull()),'Cabin']="Yes"
# data_all.loc[(data_all.Cabin.isnull()),'Cabin']="NO"
# data_all.tail(20)
# ```
# 
# 
# 
# 
# <div>
# <style>
#     .dataframe thead tr:only-child th {
#         text-align: right;
#     }
# 
#     .dataframe thead th {
#         text-align: left;
#     }
# 
#     .dataframe tbody tr th {
#         vertical-align: top;
#     }
# </style>
# <table border="1" class="dataframe">
#   <thead>
#     <tr style="text-align: right;">
#       <th></th>
#       <th>Age</th>
#       <th>Cabin</th>
#       <th>Embarked</th>
#       <th>Fare</th>
#       <th>Name</th>
#       <th>Parch</th>
#       <th>PassengerId</th>
#       <th>Pclass</th>
#       <th>Sex</th>
#       <th>SibSp</th>
#       <th>Survived</th>
#       <th>Ticket</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>1289</th>
#       <td>22.0</td>
#       <td>NO</td>
#       <td>S</td>
#       <td>7.7750</td>
#       <td>Larsson-Rondberg, Mr. Edvard A</td>
#       <td>0</td>
#       <td>1290</td>
#       <td>3</td>
#       <td>male</td>
#       <td>0</td>
#       <td>NaN</td>
#       <td>347065</td>
#     </tr>
#     <tr>
#       <th>1290</th>
#       <td>31.0</td>
#       <td>NO</td>
#       <td>Q</td>
#       <td>7.7333</td>
#       <td>Conlon, Mr. Thomas Henry</td>
#       <td>0</td>
#       <td>1291</td>
#       <td>3</td>
#       <td>male</td>
#       <td>0</td>
#       <td>NaN</td>
#       <td>21332</td>
#     </tr>
#     <tr>
#       <th>1291</th>
#       <td>30.0</td>
#       <td>Yes</td>
#       <td>S</td>
#       <td>164.8667</td>
#       <td>Bonnell, Miss. Caroline</td>
#       <td>0</td>
#       <td>1292</td>
#       <td>1</td>
#       <td>female</td>
#       <td>0</td>
#       <td>NaN</td>
#       <td>36928</td>
#     </tr>
#     <tr>
#       <th>1292</th>
#       <td>38.0</td>
#       <td>NO</td>
#       <td>S</td>
#       <td>21.0000</td>
#       <td>Gale, Mr. Harry</td>
#       <td>0</td>
#       <td>1293</td>
#       <td>2</td>
#       <td>male</td>
#       <td>1</td>
#       <td>NaN</td>
#       <td>28664</td>
#     </tr>
#     <tr>
#       <th>1293</th>
#       <td>22.0</td>
#       <td>NO</td>
#       <td>C</td>
#       <td>59.4000</td>
#       <td>Gibson, Miss. Dorothy Winifred</td>
#       <td>1</td>
#       <td>1294</td>
#       <td>1</td>
#       <td>female</td>
#       <td>0</td>
#       <td>NaN</td>
#       <td>112378</td>
#     </tr>
#     <tr>
#       <th>1294</th>
#       <td>17.0</td>
#       <td>NO</td>
#       <td>S</td>
#       <td>47.1000</td>
#       <td>Carrau, Mr. Jose Pedro</td>
#       <td>0</td>
#       <td>1295</td>
#       <td>1</td>
#       <td>male</td>
#       <td>0</td>
#       <td>NaN</td>
#       <td>113059</td>
#     </tr>
#     <tr>
#       <th>1295</th>
#       <td>43.0</td>
#       <td>Yes</td>
#       <td>C</td>
#       <td>27.7208</td>
#       <td>Frauenthal, Mr. Isaac Gerald</td>
#       <td>0</td>
#       <td>1296</td>
#       <td>1</td>
#       <td>male</td>
#       <td>1</td>
#       <td>NaN</td>
#       <td>17765</td>
#     </tr>
#     <tr>
#       <th>1296</th>
#       <td>20.0</td>
#       <td>Yes</td>
#       <td>C</td>
#       <td>13.8625</td>
#       <td>Nourney, Mr. Alfred (Baron von Drachstedt")"</td>
#       <td>0</td>
#       <td>1297</td>
#       <td>2</td>
#       <td>male</td>
#       <td>0</td>
#       <td>NaN</td>
#       <td>SC/PARIS 2166</td>
#     </tr>
#     <tr>
#       <th>1297</th>
#       <td>23.0</td>
#       <td>NO</td>
#       <td>S</td>
#       <td>10.5000</td>
#       <td>Ware, Mr. William Jeffery</td>
#       <td>0</td>
#       <td>1298</td>
#       <td>2</td>
#       <td>male</td>
#       <td>1</td>
#       <td>NaN</td>
#       <td>28666</td>
#     </tr>
#     <tr>
#       <th>1298</th>
#       <td>50.0</td>
#       <td>Yes</td>
#       <td>C</td>
#       <td>211.5000</td>
#       <td>Widener, Mr. George Dunton</td>
#       <td>1</td>
#       <td>1299</td>
#       <td>1</td>
#       <td>male</td>
#       <td>1</td>
#       <td>NaN</td>
#       <td>113503</td>
#     </tr>
#     <tr>
#       <th>1299</th>
#       <td>NaN</td>
#       <td>NO</td>
#       <td>Q</td>
#       <td>7.7208</td>
#       <td>Riordan, Miss. Johanna Hannah""</td>
#       <td>0</td>
#       <td>1300</td>
#       <td>3</td>
#       <td>female</td>
#       <td>0</td>
#       <td>NaN</td>
#       <td>334915</td>
#     </tr>
#     <tr>
#       <th>1300</th>
#       <td>3.0</td>
#       <td>NO</td>
#       <td>S</td>
#       <td>13.7750</td>
#       <td>Peacock, Miss. Treasteall</td>
#       <td>1</td>
#       <td>1301</td>
#       <td>3</td>
#       <td>female</td>
#       <td>1</td>
#       <td>NaN</td>
#       <td>SOTON/O.Q. 3101315</td>
#     </tr>
#     <tr>
#       <th>1301</th>
#       <td>NaN</td>
#       <td>NO</td>
#       <td>Q</td>
#       <td>7.7500</td>
#       <td>Naughton, Miss. Hannah</td>
#       <td>0</td>
#       <td>1302</td>
#       <td>3</td>
#       <td>female</td>
#       <td>0</td>
#       <td>NaN</td>
#       <td>365237</td>
#     </tr>
#     <tr>
#       <th>1302</th>
#       <td>37.0</td>
#       <td>Yes</td>
#       <td>Q</td>
#       <td>90.0000</td>
#       <td>Minahan, Mrs. William Edward (Lillian E Thorpe)</td>
#       <td>0</td>
#       <td>1303</td>
#       <td>1</td>
#       <td>female</td>
#       <td>1</td>
#       <td>NaN</td>
#       <td>19928</td>
#     </tr>
#     <tr>
#       <th>1303</th>
#       <td>28.0</td>
#       <td>NO</td>
#       <td>S</td>
#       <td>7.7750</td>
#       <td>Henriksson, Miss. Jenny Lovisa</td>
#       <td>0</td>
#       <td>1304</td>
#       <td>3</td>
#       <td>female</td>
#       <td>0</td>
#       <td>NaN</td>
#       <td>347086</td>
#     </tr>
#     <tr>
#       <th>1304</th>
#       <td>NaN</td>
#       <td>NO</td>
#       <td>S</td>
#       <td>8.0500</td>
#       <td>Spector, Mr. Woolf</td>
#       <td>0</td>
#       <td>1305</td>
#       <td>3</td>
#       <td>male</td>
#       <td>0</td>
#       <td>NaN</td>
#       <td>A.5. 3236</td>
#     </tr>
#     <tr>
#       <th>1305</th>
#       <td>39.0</td>
#       <td>Yes</td>
#       <td>C</td>
#       <td>108.9000</td>
#       <td>Oliva y Ocana, Dona. Fermina</td>
#       <td>0</td>
#       <td>1306</td>
#       <td>1</td>
#       <td>female</td>
#       <td>0</td>
#       <td>NaN</td>
#       <td>PC 17758</td>
#     </tr>
#     <tr>
#       <th>1306</th>
#       <td>38.5</td>
#       <td>NO</td>
#       <td>S</td>
#       <td>7.2500</td>
#       <td>Saether, Mr. Simon Sivertsen</td>
#       <td>0</td>
#       <td>1307</td>
#       <td>3</td>
#       <td>male</td>
#       <td>0</td>
#       <td>NaN</td>
#       <td>SOTON/O.Q. 3101262</td>
#     </tr>
#     <tr>
#       <th>1307</th>
#       <td>NaN</td>
#       <td>NO</td>
#       <td>S</td>
#       <td>8.0500</td>
#       <td>Ware, Mr. Frederick</td>
#       <td>0</td>
#       <td>1308</td>
#       <td>3</td>
#       <td>male</td>
#       <td>0</td>
#       <td>NaN</td>
#       <td>359309</td>
#     </tr>
#     <tr>
#       <th>1308</th>
#       <td>NaN</td>
#       <td>NO</td>
#       <td>C</td>
#       <td>22.3583</td>
#       <td>Peter, Master. Michael J</td>
#       <td>1</td>
#       <td>1309</td>
#       <td>3</td>
#       <td>male</td>
#       <td>1</td>
#       <td>NaN</td>
#       <td>2668</td>
#     </tr>
#   </tbody>
# </table>
# </div>
# 
# 
# 
# 
# ```python
# data_all=data_all.drop(['Survived','PassengerId'],axis=1)
# data_all.head()
# ```
# 
# 
# 
# 
# <div>
# <style>
#     .dataframe thead tr:only-child th {
#         text-align: right;
#     }
# 
#     .dataframe thead th {
#         text-align: left;
#     }
# 
#     .dataframe tbody tr th {
#         vertical-align: top;
#     }
# </style>
# <table border="1" class="dataframe">
#   <thead>
#     <tr style="text-align: right;">
#       <th></th>
#       <th>Age</th>
#       <th>Cabin</th>
#       <th>Embarked</th>
#       <th>Fare</th>
#       <th>Name</th>
#       <th>Parch</th>
#       <th>Pclass</th>
#       <th>Sex</th>
#       <th>SibSp</th>
#       <th>Ticket</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>0</th>
#       <td>22.0</td>
#       <td>NO</td>
#       <td>S</td>
#       <td>7.2500</td>
#       <td>Braund, Mr. Owen Harris</td>
#       <td>0</td>
#       <td>3</td>
#       <td>male</td>
#       <td>1</td>
#       <td>A/5 21171</td>
#     </tr>
#     <tr>
#       <th>1</th>
#       <td>38.0</td>
#       <td>Yes</td>
#       <td>C</td>
#       <td>71.2833</td>
#       <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
#       <td>0</td>
#       <td>1</td>
#       <td>female</td>
#       <td>1</td>
#       <td>PC 17599</td>
#     </tr>
#     <tr>
#       <th>2</th>
#       <td>26.0</td>
#       <td>NO</td>
#       <td>S</td>
#       <td>7.9250</td>
#       <td>Heikkinen, Miss. Laina</td>
#       <td>0</td>
#       <td>3</td>
#       <td>female</td>
#       <td>0</td>
#       <td>STON/O2. 3101282</td>
#     </tr>
#     <tr>
#       <th>3</th>
#       <td>35.0</td>
#       <td>Yes</td>
#       <td>S</td>
#       <td>53.1000</td>
#       <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
#       <td>0</td>
#       <td>1</td>
#       <td>female</td>
#       <td>1</td>
#       <td>113803</td>
#     </tr>
#     <tr>
#       <th>4</th>
#       <td>35.0</td>
#       <td>NO</td>
#       <td>S</td>
#       <td>8.0500</td>
#       <td>Allen, Mr. William Henry</td>
#       <td>0</td>
#       <td>3</td>
#       <td>male</td>
#       <td>0</td>
#       <td>373450</td>
#     </tr>
#   </tbody>
# </table>
# </div>
# 
# 
# 
# 
# ```python
# data_all['Cabin'].tail(20)
# ```
# 
# 
# 
# 
#     1289     NO
#     1290     NO
#     1291    Yes
#     1292     NO
#     1293     NO
#     1294     NO
#     1295    Yes
#     1296    Yes
#     1297     NO
#     1298    Yes
#     1299     NO
#     1300     NO
#     1301     NO
#     1302    Yes
#     1303     NO
#     1304     NO
#     1305    Yes
#     1306     NO
#     1307     NO
#     1308     NO
#     Name: Cabin, dtype: object
# 
# 
# 
# 
# ```python
# data_train.head(10)
# ```
# 
# 
# 
# 
# <div>
# <style>
#     .dataframe thead tr:only-child th {
#         text-align: right;
#     }
# 
#     .dataframe thead th {
#         text-align: left;
#     }
# 
#     .dataframe tbody tr th {
#         vertical-align: top;
#     }
# </style>
# <table border="1" class="dataframe">
#   <thead>
#     <tr style="text-align: right;">
#       <th></th>
#       <th>PassengerId</th>
#       <th>Survived</th>
#       <th>Pclass</th>
#       <th>Name</th>
#       <th>Sex</th>
#       <th>Age</th>
#       <th>SibSp</th>
#       <th>Parch</th>
#       <th>Ticket</th>
#       <th>Fare</th>
#       <th>Cabin</th>
#       <th>Embarked</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>0</th>
#       <td>1</td>
#       <td>0</td>
#       <td>3</td>
#       <td>Braund, Mr. Owen Harris</td>
#       <td>male</td>
#       <td>22.0</td>
#       <td>1</td>
#       <td>0</td>
#       <td>A/5 21171</td>
#       <td>7.2500</td>
#       <td>NO</td>
#       <td>S</td>
#     </tr>
#     <tr>
#       <th>1</th>
#       <td>2</td>
#       <td>1</td>
#       <td>1</td>
#       <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
#       <td>female</td>
#       <td>38.0</td>
#       <td>1</td>
#       <td>0</td>
#       <td>PC 17599</td>
#       <td>71.2833</td>
#       <td>Yes</td>
#       <td>C</td>
#     </tr>
#     <tr>
#       <th>2</th>
#       <td>3</td>
#       <td>1</td>
#       <td>3</td>
#       <td>Heikkinen, Miss. Laina</td>
#       <td>female</td>
#       <td>26.0</td>
#       <td>0</td>
#       <td>0</td>
#       <td>STON/O2. 3101282</td>
#       <td>7.9250</td>
#       <td>NO</td>
#       <td>S</td>
#     </tr>
#     <tr>
#       <th>3</th>
#       <td>4</td>
#       <td>1</td>
#       <td>1</td>
#       <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
#       <td>female</td>
#       <td>35.0</td>
#       <td>1</td>
#       <td>0</td>
#       <td>113803</td>
#       <td>53.1000</td>
#       <td>Yes</td>
#       <td>S</td>
#     </tr>
#     <tr>
#       <th>4</th>
#       <td>5</td>
#       <td>0</td>
#       <td>3</td>
#       <td>Allen, Mr. William Henry</td>
#       <td>male</td>
#       <td>35.0</td>
#       <td>0</td>
#       <td>0</td>
#       <td>373450</td>
#       <td>8.0500</td>
#       <td>NO</td>
#       <td>S</td>
#     </tr>
#     <tr>
#       <th>5</th>
#       <td>6</td>
#       <td>0</td>
#       <td>3</td>
#       <td>Moran, Mr. James</td>
#       <td>male</td>
#       <td>NaN</td>
#       <td>0</td>
#       <td>0</td>
#       <td>330877</td>
#       <td>8.4583</td>
#       <td>NO</td>
#       <td>Q</td>
#     </tr>
#     <tr>
#       <th>6</th>
#       <td>7</td>
#       <td>0</td>
#       <td>1</td>
#       <td>McCarthy, Mr. Timothy J</td>
#       <td>male</td>
#       <td>54.0</td>
#       <td>0</td>
#       <td>0</td>
#       <td>17463</td>
#       <td>51.8625</td>
#       <td>Yes</td>
#       <td>S</td>
#     </tr>
#     <tr>
#       <th>7</th>
#       <td>8</td>
#       <td>0</td>
#       <td>3</td>
#       <td>Palsson, Master. Gosta Leonard</td>
#       <td>male</td>
#       <td>2.0</td>
#       <td>3</td>
#       <td>1</td>
#       <td>349909</td>
#       <td>21.0750</td>
#       <td>NO</td>
#       <td>S</td>
#     </tr>
#     <tr>
#       <th>8</th>
#       <td>9</td>
#       <td>1</td>
#       <td>3</td>
#       <td>Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)</td>
#       <td>female</td>
#       <td>27.0</td>
#       <td>0</td>
#       <td>2</td>
#       <td>347742</td>
#       <td>11.1333</td>
#       <td>NO</td>
#       <td>S</td>
#     </tr>
#     <tr>
#       <th>9</th>
#       <td>10</td>
#       <td>1</td>
#       <td>2</td>
#       <td>Nasser, Mrs. Nicholas (Adele Achem)</td>
#       <td>female</td>
#       <td>14.0</td>
#       <td>1</td>
#       <td>0</td>
#       <td>237736</td>
#       <td>30.0708</td>
#       <td>NO</td>
#       <td>C</td>
#     </tr>
#   </tbody>
# </table>
# </div>
# 
# 
# 
# 
# ```python
# data_all.isnull().sum()
# ```
# 
# 
# 
# 
#     Age         263
#     Cabin         0
#     Embarked      2
#     Fare          1
#     Name          0
#     Parch         0
#     Pclass        0
#     Sex           0
#     SibSp         0
#     Ticket        0
#     dtype: int64
# 
# 
# 
# 
# ```python
# data_all['Pclass'][data_all['Fare'].isnull()]
# ```
# 
# 
# 
# 
#     1043    3
#     Name: Pclass, dtype: int64
# 
# 
# 
# 
# ```python
# data_all['Fare'][data_all.Pclass.values==3].mean()
# ```
# 
# 
# 
# 
#     13.302888700564969
# 
# 
# 
# 
# ```python
# data_all['Fare'].fillna(data_all['Fare'][data_all.Pclass.values==3].mean())
# ```
# 
# 
# 
# 
#     0         7.2500
#     1        71.2833
#     2         7.9250
#     3        53.1000
#     4         8.0500
#     5         8.4583
#     6        51.8625
#     7        21.0750
#     8        11.1333
#     9        30.0708
#     10       16.7000
#     11       26.5500
#     12        8.0500
#     13       31.2750
#     14        7.8542
#     15       16.0000
#     16       29.1250
#     17       13.0000
#     18       18.0000
#     19        7.2250
#     20       26.0000
#     21       13.0000
#     22        8.0292
#     23       35.5000
#     24       21.0750
#     25       31.3875
#     26        7.2250
#     27      263.0000
#     28        7.8792
#     29        7.8958
#               ...   
#     1279      7.7500
#     1280     21.0750
#     1281     93.5000
#     1282     39.4000
#     1283     20.2500
#     1284     10.5000
#     1285     22.0250
#     1286     60.0000
#     1287      7.2500
#     1288     79.2000
#     1289      7.7750
#     1290      7.7333
#     1291    164.8667
#     1292     21.0000
#     1293     59.4000
#     1294     47.1000
#     1295     27.7208
#     1296     13.8625
#     1297     10.5000
#     1298    211.5000
#     1299      7.7208
#     1300     13.7750
#     1301      7.7500
#     1302     90.0000
#     1303      7.7750
#     1304      8.0500
#     1305    108.9000
#     1306      7.2500
#     1307      8.0500
#     1308     22.3583
#     Name: Fare, Length: 1309, dtype: float64
# 
# 
# 
# 
# ```python
# data_all.isnull().sum()
# ```
# 
# 
# 
# 
#     Age         263
#     Cabin         0
#     Embarked      2
#     Fare          1
#     Name          0
#     Parch         0
#     Pclass        0
#     Sex           0
#     SibSp         0
#     Ticket        0
#     dtype: int64
# 
# 
# 
# 
# ```python
# data_all['Fare'].fillna(data_all['Fare'][data_all.Pclass.values==3].mean(),inplace=True)
# ```
# 
# 
# ```python
# data_all.isnull().sum()
# ```
# 
# 
# 
# 
#     Age         263
#     Cabin         0
#     Embarked      2
#     Fare          0
#     Name          0
#     Parch         0
#     Pclass        0
#     Sex           0
#     SibSp         0
#     Ticket        0
#     dtype: int64
# 
# 
# 
# 
# ```python
# data_all['Embarked'].value_counts()
# ```
# 
# 
# 
# 
#     S    914
#     C    270
#     Q    123
#     Name: Embarked, dtype: int64
# 
# 
# 
# 
# ```python
# data_all['Embarked'].fillna('S',inplace=True)
# ```
# 
# 
# ```python
# data_all.head(10)
# ```
# 
# 
# 
# 
# <div>
# <style>
#     .dataframe thead tr:only-child th {
#         text-align: right;
#     }
# 
#     .dataframe thead th {
#         text-align: left;
#     }
# 
#     .dataframe tbody tr th {
#         vertical-align: top;
#     }
# </style>
# <table border="1" class="dataframe">
#   <thead>
#     <tr style="text-align: right;">
#       <th></th>
#       <th>Age</th>
#       <th>Cabin</th>
#       <th>Embarked</th>
#       <th>Fare</th>
#       <th>Name</th>
#       <th>Parch</th>
#       <th>Pclass</th>
#       <th>Sex</th>
#       <th>SibSp</th>
#       <th>Ticket</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>0</th>
#       <td>22.0</td>
#       <td>NO</td>
#       <td>S</td>
#       <td>7.2500</td>
#       <td>Braund, Mr. Owen Harris</td>
#       <td>0</td>
#       <td>3</td>
#       <td>male</td>
#       <td>1</td>
#       <td>A/5 21171</td>
#     </tr>
#     <tr>
#       <th>1</th>
#       <td>38.0</td>
#       <td>Yes</td>
#       <td>C</td>
#       <td>71.2833</td>
#       <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
#       <td>0</td>
#       <td>1</td>
#       <td>female</td>
#       <td>1</td>
#       <td>PC 17599</td>
#     </tr>
#     <tr>
#       <th>2</th>
#       <td>26.0</td>
#       <td>NO</td>
#       <td>S</td>
#       <td>7.9250</td>
#       <td>Heikkinen, Miss. Laina</td>
#       <td>0</td>
#       <td>3</td>
#       <td>female</td>
#       <td>0</td>
#       <td>STON/O2. 3101282</td>
#     </tr>
#     <tr>
#       <th>3</th>
#       <td>35.0</td>
#       <td>Yes</td>
#       <td>S</td>
#       <td>53.1000</td>
#       <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
#       <td>0</td>
#       <td>1</td>
#       <td>female</td>
#       <td>1</td>
#       <td>113803</td>
#     </tr>
#     <tr>
#       <th>4</th>
#       <td>35.0</td>
#       <td>NO</td>
#       <td>S</td>
#       <td>8.0500</td>
#       <td>Allen, Mr. William Henry</td>
#       <td>0</td>
#       <td>3</td>
#       <td>male</td>
#       <td>0</td>
#       <td>373450</td>
#     </tr>
#     <tr>
#       <th>5</th>
#       <td>NaN</td>
#       <td>NO</td>
#       <td>Q</td>
#       <td>8.4583</td>
#       <td>Moran, Mr. James</td>
#       <td>0</td>
#       <td>3</td>
#       <td>male</td>
#       <td>0</td>
#       <td>330877</td>
#     </tr>
#     <tr>
#       <th>6</th>
#       <td>54.0</td>
#       <td>Yes</td>
#       <td>S</td>
#       <td>51.8625</td>
#       <td>McCarthy, Mr. Timothy J</td>
#       <td>0</td>
#       <td>1</td>
#       <td>male</td>
#       <td>0</td>
#       <td>17463</td>
#     </tr>
#     <tr>
#       <th>7</th>
#       <td>2.0</td>
#       <td>NO</td>
#       <td>S</td>
#       <td>21.0750</td>
#       <td>Palsson, Master. Gosta Leonard</td>
#       <td>1</td>
#       <td>3</td>
#       <td>male</td>
#       <td>3</td>
#       <td>349909</td>
#     </tr>
#     <tr>
#       <th>8</th>
#       <td>27.0</td>
#       <td>NO</td>
#       <td>S</td>
#       <td>11.1333</td>
#       <td>Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)</td>
#       <td>2</td>
#       <td>3</td>
#       <td>female</td>
#       <td>0</td>
#       <td>347742</td>
#     </tr>
#     <tr>
#       <th>9</th>
#       <td>14.0</td>
#       <td>NO</td>
#       <td>C</td>
#       <td>30.0708</td>
#       <td>Nasser, Mrs. Nicholas (Adele Achem)</td>
#       <td>0</td>
#       <td>2</td>
#       <td>female</td>
#       <td>1</td>
#       <td>237736</td>
#     </tr>
#   </tbody>
# </table>
# </div>
# 
# 
# 
# 
# ```python
# data_all.isnull().sum()
# ```
# 
# 
# 
# 
#     Age         0
#     Cabin       0
#     Embarked    0
#     Fare        0
#     Name        0
#     Parch       0
#     Pclass      0
#     Sex         0
#     SibSp       0
#     Ticket      0
#     dtype: int64
# 
# 
# 
# 
# ```python
# data_all['Age'].fillna((data_all['Age'].mean()),inplace=True)
# ```
# 
# 
# ```python
# data_all['Family']=data_all['Parch']+data_all['SibSp']
# ```
# 
# 
# ```python
# data_all.head()
# ```
# 
# 
# 
# 
# <div>
# <style>
#     .dataframe thead tr:only-child th {
#         text-align: right;
#     }
# 
#     .dataframe thead th {
#         text-align: left;
#     }
# 
#     .dataframe tbody tr th {
#         vertical-align: top;
#     }
# </style>
# <table border="1" class="dataframe">
#   <thead>
#     <tr style="text-align: right;">
#       <th></th>
#       <th>Age</th>
#       <th>Fare</th>
#       <th>Pclass</th>
#       <th>Family</th>
#       <th>Cabin_NO</th>
#       <th>Cabin_Yes</th>
#       <th>Embarked_C</th>
#       <th>Embarked_Q</th>
#       <th>Embarked_S</th>
#       <th>Sex_female</th>
#       <th>Sex_male</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>0</th>
#       <td>22.0</td>
#       <td>7.2500</td>
#       <td>3</td>
#       <td>1</td>
#       <td>1</td>
#       <td>0</td>
#       <td>0</td>
#       <td>0</td>
#       <td>1</td>
#       <td>0</td>
#       <td>1</td>
#     </tr>
#     <tr>
#       <th>1</th>
#       <td>38.0</td>
#       <td>71.2833</td>
#       <td>1</td>
#       <td>1</td>
#       <td>0</td>
#       <td>1</td>
#       <td>1</td>
#       <td>0</td>
#       <td>0</td>
#       <td>1</td>
#       <td>0</td>
#     </tr>
#     <tr>
#       <th>2</th>
#       <td>26.0</td>
#       <td>7.9250</td>
#       <td>3</td>
#       <td>0</td>
#       <td>1</td>
#       <td>0</td>
#       <td>0</td>
#       <td>0</td>
#       <td>1</td>
#       <td>1</td>
#       <td>0</td>
#     </tr>
#     <tr>
#       <th>3</th>
#       <td>35.0</td>
#       <td>53.1000</td>
#       <td>1</td>
#       <td>1</td>
#       <td>0</td>
#       <td>1</td>
#       <td>0</td>
#       <td>0</td>
#       <td>1</td>
#       <td>1</td>
#       <td>0</td>
#     </tr>
#     <tr>
#       <th>4</th>
#       <td>35.0</td>
#       <td>8.0500</td>
#       <td>3</td>
#       <td>0</td>
#       <td>1</td>
#       <td>0</td>
#       <td>0</td>
#       <td>0</td>
#       <td>1</td>
#       <td>0</td>
#       <td>1</td>
#     </tr>
#   </tbody>
# </table>
# </div>
# 
# 
# 
# 
# ```python
# data_all=data_all.drop(['Parch','Name','SibSp','Ticket'],axis=1)
# ```
# 
# 
# ```python
# data_all=pd.get_dummies(data_all)
# ```
# 
# 
# ```python
# X_train=data_all[:data_train.shape[0]]
# X_test=data_all[data_train.shape[0]:]
# y_train=y
# ```
# 
# 
# ```python
# X_train.info()
# X_test.info()
# y
# ```
# 
#     <class 'pandas.core.frame.DataFrame'>
#     RangeIndex: 891 entries, 0 to 890
#     Data columns (total 11 columns):
#     Age           891 non-null float64
#     Fare          891 non-null float64
#     Pclass        891 non-null int64
#     Family        891 non-null int64
#     Cabin_NO      891 non-null uint8
#     Cabin_Yes     891 non-null uint8
#     Embarked_C    891 non-null uint8
#     Embarked_Q    891 non-null uint8
#     Embarked_S    891 non-null uint8
#     Sex_female    891 non-null uint8
#     Sex_male      891 non-null uint8
#     dtypes: float64(2), int64(2), uint8(7)
#     memory usage: 34.0 KB
#     <class 'pandas.core.frame.DataFrame'>
#     RangeIndex: 418 entries, 891 to 1308
#     Data columns (total 11 columns):
#     Age           418 non-null float64
#     Fare          418 non-null float64
#     Pclass        418 non-null int64
#     Family        418 non-null int64
#     Cabin_NO      418 non-null uint8
#     Cabin_Yes     418 non-null uint8
#     Embarked_C    418 non-null uint8
#     Embarked_Q    418 non-null uint8
#     Embarked_S    418 non-null uint8
#     Sex_female    418 non-null uint8
#     Sex_male      418 non-null uint8
#     dtypes: float64(2), int64(2), uint8(7)
#     memory usage: 16.0 KB
#     
# 
# 
# 
# 
#     0      0
#     1      1
#     2      1
#     3      1
#     4      0
#     5      0
#     6      0
#     7      0
#     8      1
#     9      1
#     10     1
#     11     1
#     12     0
#     13     0
#     14     0
#     15     1
#     16     0
#     17     1
#     18     0
#     19     1
#     20     0
#     21     1
#     22     1
#     23     1
#     24     0
#     25     1
#     26     0
#     27     0
#     28     1
#     29     0
#           ..
#     861    0
#     862    1
#     863    0
#     864    0
#     865    1
#     866    1
#     867    0
#     868    0
#     869    1
#     870    0
#     871    1
#     872    0
#     873    0
#     874    1
#     875    1
#     876    0
#     877    0
#     878    0
#     879    1
#     880    1
#     881    0
#     882    0
#     883    0
#     884    0
#     885    0
#     886    0
#     887    1
#     888    0
#     889    1
#     890    0
#     Name: Survived, Length: 891, dtype: int64
# 
# 
# 
# 
# ```python
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
# from sklearn.svm import SVC
# from sklearn.cross_validation import KFold;
# ```
# 
# 
# ```python
# rf=RandomForestClassifier()
# ```
# 
# 
# ```python
# rf=rf.fit(X_train,y)
# ```
# 
# 
# ```python
# y_test=rf.predict(X_test)
# ```
# 
# 
# ```python
# y_test
# ```
# 
# 
# 
# 
#     array([0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1,
#            0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0,
#            0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0,
#            1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0,
#            1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1,
#            0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#            0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1,
#            1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0,
#            1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1,
#            0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
#            0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0,
#            0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
#            0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
#            0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0,
#            1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
#            0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0,
#            1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0,
#            1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
#            1, 0, 0, 0], dtype=int64)
# 
# 
# 
# 
# ```python
# #titanic_test1_result=pd.DataFrame([data_test['PassengerId'],y_test],columns=['PassengerId','Survived'])
# titanic_test1_result=pd.DataFrame({ 'PassengerId': data_test.PassengerId, 'Survived': y_test })
# ```
# 
# 
# ```python
# titanic_test1_result.head()
# ```
# 
# 
# 
# 
# <div>
# <style>
#     .dataframe thead tr:only-child th {
#         text-align: right;
#     }
# 
#     .dataframe thead th {
#         text-align: left;
#     }
# 
#     .dataframe tbody tr th {
#         vertical-align: top;
#     }
# </style>
# <table border="1" class="dataframe">
#   <thead>
#     <tr style="text-align: right;">
#       <th></th>
#       <th>PassengerId</th>
#       <th>Survived</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>0</th>
#       <td>892</td>
#       <td>0</td>
#     </tr>
#     <tr>
#       <th>1</th>
#       <td>893</td>
#       <td>0</td>
#     </tr>
#     <tr>
#       <th>2</th>
#       <td>894</td>
#       <td>0</td>
#     </tr>
#     <tr>
#       <th>3</th>
#       <td>895</td>
#       <td>1</td>
#     </tr>
#     <tr>
#       <th>4</th>
#       <td>896</td>
#       <td>1</td>
#     </tr>
#   </tbody>
# </table>
# </div>
# 
# 
# 
# 
# ```python
# titanic_test1_result.to_csv("D:\ML_Data\Tianic\Titanic_test1_result.csv", index=False)
# ```
# 
