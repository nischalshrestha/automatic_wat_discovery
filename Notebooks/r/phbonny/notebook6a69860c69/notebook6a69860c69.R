# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



# Any results you write to the current directory are saved as output.
library(h2o)

library(statmod)
c1 <- h2o.init()
df <- h2o.importFile(path ="../input/train.csv")
df
response <- "Survived"

df[[response]] <- as.factor(df[[response]])

predictors <- setdiff(names(df), c(response, "Name"))

summary(df,exact_quantiles=TRUE)
splits <- h2o.splitFrame(

  data = df, 

  ratios = c(0.6,0.2),   ## only need to specify 2 fractions, the 3rd is implied

  destination_frames = c("train.hex", "valid.hex", "test.hex"), seed = 1234

)

train <- splits[[1]]

valid <- splits[[2]]

test  <- splits[[3]]
## We only provide the required parameters, everything else is default

gbm <- h2o.gbm(x = predictors, y = response, training_frame = train)



## Show a detailed model summary

gbm



## Get the AUC on the validation set

h2o.auc(h2o.performance(gbm, newdata = valid)) 
## New model

## h2o.rbind makes a copy here, so it's better to use splitFrame with `ratios = c(0.8)` instead above

gbm <- h2o.gbm(x = predictors, y = response, training_frame = h2o.rbind(train, valid), nfolds = 5, seed = 0xDECAF)



## Show a detailed summary of the cross validation metrics

## This gives you an idea of the variance between the folds

gbm@model$cross_validation_metrics_summary



## Get the cross-validated AUC by scoring the combined holdout predictions.

## (Instead of taking the average of the metrics across the folds)

h2o.auc(h2o.performance(gbm, xval = TRUE))
## New model

gbm <- h2o.gbm(

  ## standard model parameters

  x = predictors, 

  y = response, 

  training_frame = train, 

  validation_frame = valid,

  

  ## more trees is better if the learning rate is small enough 

  ## here, use "more than enough" trees - we have early stopping

  ntrees = 10000,                                                            

  

  ## smaller learning rate is better (this is a good value for most datasets, but see below for annealing)

  learn_rate=0.01,                                                         

  

  ## early stopping once the validation AUC doesn't improve by at least 0.01% for 5 consecutive scoring events

  stopping_rounds = 5, stopping_tolerance = 1e-4, stopping_metric = "AUC", 

  

  ## sample 80% of rows per tree

  sample_rate = 0.8,                                                       



  ## sample 80% of columns per split

  col_sample_rate = 0.8,                                                   



  ## fix a random number generator seed for reproducibility

  seed = 1234,                                                             

  

  ## score every 10 trees to make early stopping reproducible (it depends on the scoring interval)

  score_tree_interval = 10                                                 

)



## Get the AUC on the validation set

h2o.auc(h2o.performance(gbm, valid = TRUE))
## Depth 10 is usually plenty of depth for most datasets, but you never know

hyper_params = list( ntrees=c(10,100,1000, 10000), max_depth = seq(1,29,2) )

#hyper_params = list( max_depth = c(4,6,8,12,16,20) ) ##faster for larger datasets



grid <- h2o.grid(

  ## hyper parameters

  hyper_params = hyper_params,

  

  ## full Cartesian hyper-parameter search

  search_criteria = list(strategy = "Cartesian"),

  

  ## which algorithm to run

  algorithm="gbm",

  

  ## identifier for the grid, to later retrieve it

  grid_id="depth_grid",

  

  ## standard model parameters

  x = predictors, 

  y = response, 

  training_frame = train, 

  validation_frame = valid,

  

  ## more trees is better if the learning rate is small enough 

  ## here, use "more than enough" trees - we have early stopping

  #ntrees = 10000,                                                            

  

  ## smaller learning rate is better

  ## since we have learning_rate_annealing, we can afford to start with a bigger learning rate

  learn_rate = 0.05,                                                         

  

  ## learning rate annealing: learning_rate shrinks by 1% after every tree 

  ## (use 1.00 to disable, but then lower the learning_rate)

  learn_rate_annealing = 0.99,                                               

  

  ## sample 80% of rows per tree

  sample_rate = 0.8,                                                       



  ## sample 80% of columns per split

  col_sample_rate = 0.8, 

  

  ## fix a random number generator seed for reproducibility

  seed = 1234,                                                             

  

  ## early stopping once the validation AUC doesn't improve by at least 0.01% for 5 consecutive scoring events

  stopping_rounds = 5,

  stopping_tolerance = 1e-4,

  stopping_metric = "AUC", 

  

  ## score every 10 trees to make early stopping reproducible (it depends on the scoring interval)

  score_tree_interval = 10                                                

)



## by default, display the grid search results sorted by increasing logloss (since this is a classification task)

grid                                                                       



## sort the grid models by decreasing AUC

sortedGrid <- h2o.getGrid("depth_grid", sort_by="auc", decreasing = TRUE)    

sortedGrid



## find the range of max_depth for the top 5 models

topDepths = sortedGrid@summary_table$max_depth[1:5]                       

minDepth = min(as.numeric(topDepths))

maxDepth = max(as.numeric(topDepths))
maxDepth