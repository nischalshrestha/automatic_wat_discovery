#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

# feature_columns_to_use
fctu = ["Pclass", "Sex", "Age", "Fare", "Parch"]

big_X = train_df[fctu].append(test_df[fctu])

# 处理缺失值
big_X['Fare'] = big_X['Fare'].fillna(big_X['Fare'].median())

# 处理Age:统计各Age出现的频率
# 统计Age出现的频数,得到的索引才是Age,而Age已经变成了相应的频数
ages_prob = big_X['Age'].value_counts().to_frame()
# 调整下名称,Age->Count,索引->Age
ages_prob['real_age'] = ages_prob.index
ages_prob = ages_prob.rename(columns={"Age": "Count", "real_age": "Age"})
# 调整列的前后顺序,使Age在前
ages_prob = ages_prob.reindex(columns=["Age", "Count"])
# 重置索引,使不再以年龄为索引
ages_prob = ages_prob.reset_index()
# 删除上一步多出来的index列
ages_prob = ages_prob.drop(["index"], axis=1)
# 增加频率一列:频率=频数/总数
ages_prob["Prob"] = ages_prob["Count"] / big_X["Age"].value_counts().sum()

# 对缺失年龄的处理:从ages列表中以概率列表probs随机选择和缺失值个数一样多的Age形成newAges
ages = ages_prob["Age"].values.tolist()
probs = ages_prob["Prob"].values.tolist()
newAges = np.random.choice(ages, big_X['Age'].isnull().sum(), probs)
# 填充进big_X中的缺失位置
AgeNulls = big_X[pd.isnull(big_X['Age'])]
for i, ni in enumerate(AgeNulls.index[:len(newAges)]):
    big_X.loc[ni, ['Age']] = newAges[i]

big_X_inputed = big_X.copy()

# XGBoost不能自动处理类别特征,所以转换成integer类型的列
le = LabelEncoder()  # 对标签进行编码
big_X_inputed['Sex'] = le.fit_transform(big_X_inputed['Sex'])

# 再把big_X_inputed(因为顺序没变)分回训练集和测试集
train_X = big_X_inputed[0:train_df.shape[0]].values
test_X = big_X_inputed[train_df.shape[0]::].values
train_y = train_df['Survived']

# XGBoost的训练和预测
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)
predictions = gbm.predict(test_X)
print(gbm.score(train_X, train_y))

# Kaggle提交的格式
submission = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": predictions})
submission.to_csv("post.csv", index=False)

