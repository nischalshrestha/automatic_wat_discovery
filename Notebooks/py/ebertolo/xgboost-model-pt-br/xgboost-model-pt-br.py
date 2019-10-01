#!/usr/bin/env python
# coding: utf-8

# # Titanic - XGBoost
# Este notebook cria um modelo baseado no dataset do Titanic e usando XGBoost.

# Vamos começar importando as bibliotecas básicas que vamos usar.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# Próximo passo: carregando os dados a partir dos CSVs disponibilizados no Kaggle. Estamos usando a biblioteca pandas para esse propósito.

# In[ ]:


# Vamos iniciar o notebook importanto o Dataset
titanic_df = pd.read_csv("../input/train.csv")
test_df    = pd.read_csv("../input/test.csv")

# Podemos observar as primeiras linhas dele.
titanic_df.head()


# Vamos começar com o básico de tratamento desse dataset. Importante: tudo que fizermos vamos fazer no dataset de treinamento e também de teste.

# ## Tratando a Idade - Imputation

# Teremos que preencher isso de algum jeito. Uma abordagem comum nesses casos é usar uma média ou mediana. Vamos usar aqui a mediana do dataset - mas poderíamos agrupar por sexo, por exemplo. Fica a seu critério fazer isso de forma mais fancy. ;)

# In[ ]:


age_median = titanic_df['Age'].median()
print(age_median)


# In[ ]:


titanic_df['Age'] = titanic_df['Age'].fillna(age_median)
test_df['Age'] = test_df['Age'].fillna(age_median)


# ## Tratando Gênero - LabelEncoding

# In[ ]:


from sklearn.preprocessing import LabelEncoder
sex_encoder = LabelEncoder()

sex_encoder.fit(list(titanic_df['Sex'].values) + list(test_df['Sex'].values))


# In[ ]:


sex_encoder.classes_


# In[ ]:


titanic_df['Sex'] = sex_encoder.transform(titanic_df['Sex'].values)
test_df['Sex'] = sex_encoder.transform(test_df['Sex'].values)


# ## Feature Engineering - Título

# Feature Engineering é uma técnica que envolve criar novas features - em geral a partir de outras. Vamos usar essa técnica para extrair o título a partir do nome.

# In[ ]:


titanic_df.head()['Name']


# In[ ]:


import re
def extract_title(name):
    x = re.search(', (.+)\.', name)
    if x:
        return x.group(1)
    else:
        return ''


# In[ ]:


titanic_df['Title'] = titanic_df['Name'].apply(extract_title)
test_df['Title'] = test_df['Name'].apply(extract_title)


# ## OneHotEncoding

# Agora vamos trabalhar com features que são MultiCategoricas. 

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer

feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Title', 'Embarked']
dv = DictVectorizer()
dv.fit(titanic_df[feature_names].append(test_df[feature_names]).to_dict(orient='records'))
dv.feature_names_


# In[ ]:


from sklearn.model_selection import train_test_split
train_X, valid_X, train_y, valid_y = train_test_split(dv.transform(titanic_df[feature_names].to_dict(orient='records')),
                                                     titanic_df['Survived'],
                                                     test_size=0.2,
                                                     random_state=42)


# In[ ]:


import xgboost as xgb


# In[ ]:


train_X.todense()


# In[ ]:


dtrain = xgb.DMatrix(data=train_X.todense(), feature_names=dv.feature_names_, label=train_y)
dvalid = xgb.DMatrix(data=valid_X.todense(), feature_names=dv.feature_names_, label=valid_y)


# In[ ]:


xgb_clf = xgb.train({'max_depth':20, 'eta':0.1, 'objective':'binary:logistic', 'eval_metric': 'error'}, 
                    num_boost_round=3000,
                    dtrain=dtrain,
                    verbose_eval=True, 
                    early_stopping_rounds=30,
                    evals=[(dtrain, 'train'), (dvalid, 'valid')])


# In[ ]:


from xgboost import plot_tree
ax = plot_tree(xgb_clf, num_trees=xgb_clf.best_ntree_limit-1)
ax.figure.set_size_inches((30,40))


# ## Submissão do Arquivo

# In[ ]:


test_df['Fare'] = test_df['Fare'].fillna(0)


# Lembra que o sklean trabalha com matrizes numpy, certo?

# In[ ]:


test_X = dv.transform(test_df[feature_names].to_dict(orient='records'))
print(test_X.shape)


# In[ ]:


dtest = xgb.DMatrix(data=test_X.todense(), feature_names=dv.feature_names_)


# In[ ]:


y_pred = np.round(xgb_clf.predict(dtest)).astype(int)


# Ótimo! Já temos aquilo que precisávamos. Próximo passo agora é empacotar num arquivo CSV e submeter no Kaggle.

# In[ ]:


submission_df = pd.DataFrame()


# In[ ]:


submission_df['PassengerId'] = test_df['PassengerId']
submission_df['Survived'] = y_pred
submission_df


# In[ ]:


submission_df.to_csv('xgboost_model.csv', index=False)


# Por favor, anote aqui para referência: quanto foi o seu score de treinamento do modelo? E no dataset de Validação? Quanto foi o seu score na submissão do Kaggle?

# In[ ]:




