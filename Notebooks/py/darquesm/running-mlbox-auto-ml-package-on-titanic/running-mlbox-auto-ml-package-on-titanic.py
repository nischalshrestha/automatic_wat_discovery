#!/usr/bin/env python
# coding: utf-8

# Testing the brand new Python package for Auto Machine Learning by Axel de Romblay
# 
# **https://github.com/AxeldeRomblay/MLBox**
# 
# Very promising

# # Inputs & imports : that's all you need to give !

# In[ ]:


from mlbox.preprocessing import *
from mlbox.optimisation import *
from mlbox.prediction import *


# In[ ]:


paths = ["../input/train.csv","../input/test.csv"]
target_name = "Survived"


# # Now let MLBox do the job ! 

# ## ... to read and clean all the files 

# In[ ]:


time.sleep(30)   #waiting for the engines to start


# In[ ]:


rd = Reader(sep = ",")
df = rd.train_test_split(paths, target_name)   #reading and preprocessing (dates, ...)


# In[ ]:


dft = Drift_thresholder()
df = dft.fit_transform(df)   #removing non-stable features (like ID,...)


# ## ... to tune all the hyper-parameters

# In[ ]:


opt = Optimiser(scoring = "accuracy", n_folds = 5)


# **LightGBM**

# In[ ]:


space = {
    
        'est__strategy':{"search":"choice",
                                  "space":["LightGBM"]},    
        'est__n_estimators':{"search":"choice",
                                  "space":[150]},    
        'est__colsample_bytree':{"search":"uniform",
                                  "space":[0.8,0.95]},
        'est__subsample':{"search":"uniform",
                                  "space":[0.8,0.95]},
        'est__max_depth':{"search":"choice",
                                  "space":[5,6,7,8,9]},
        'est__learning_rate':{"search":"choice",
                                  "space":[0.07]} 
    
        }

params = opt.optimise(space, df,15)


# But you can also tune the whole Pipeline ! Indeed, you can choose:
# 
# * different strategies to impute missing values
# * different strategies to encode categorical features (entity embeddings, ...)
# * different strategies and thresholds to select relevant features (random forest feature importance, l1 regularization, ...)
# * to add stacking meta-features !
# * different models and hyper-parameters (XGBoost, Random Forest, Linear, ...)

# ## ... to predict

# In[ ]:


prd = Predictor()
prd.fit_predict(params, df)


# ### Formatting for submission

# In[ ]:


submit = pd.read_csv("../input/gendermodel.csv",sep=',')
preds = pd.read_csv("save/"+target_name+"_predictions.csv")

submit[target_name] =  preds[target_name+"_predicted"].values

submit.to_csv("mlbox.csv", index=False)


# # That's all !!
# 
# If you like Axel's new auto-ml package, please **put a star on github and fork/vote the Kaggle script :)**
