#load packages

require(dplyr)

require(plyr)

require(mice)

require(caret)

require(ranger)

require(e1071)

require(xgboost)

require(kernlab)

require(klaR)
#get data

train=read.table('../input/train.csv', sep=',', header=T)

train.stop=nrow(train)

test=read.table('../input/test.csv', sep=',', header=T)

data=bind_rows(train, test)

data$Survived=as.factor(as.character(data$Survived))

data$Pclass=as.factor(as.character(data$Pclass))

data$Embarked=as.factor(as.character(data$Embarked))

data$Cabin=as.factor(as.character(data$Cabin))
data[!is.na(data) & data =='']=NA

colSums(is.na(data))

data[is.na(data$Age),'Age']=median(data$Age, na.rm=T)

data[is.na(data$Fare),'Fare']=tapply(data$Fare, data$Pclass, function(x) median(x, na.rm=T))[data[is.na(data$Fare),'Pclass']]

data[data$Fare==0,'Fare']=tapply(data$Fare, data$Pclass, function(x) median(x, na.rm=T))[data[data$Fare==0,'Pclass']]

data[is.na(data$Embarked),'Embarked']=as.factor(names(which.max(table(na.omit(data$Embarked)))))
# a function that compares different classification methods

compareML <- function(data, formula=Survived~Sex+Age, seed=123, k=10, ...){

  set.seed(seed)

  myControl <- trainControl(method = "cv", number = k,repeats = k, verboseIter = FALSE)

  glm_model <- train(formula, data = data, method="glm", family="binomial", trControl = myControl) #logreg

  rf_model <- train(formula, data = data, method = "ranger", trControl = myControl, importance = 'impurity') #rf

  glmnet_model <- train(formula, method = "glmnet", tuneGrid = expand.grid(alpha = 0:1,lambda = seq(0.0001, 1, length = 20)), data = data, trControl=myControl)  # elastic net

  xgb.grid <- expand.grid(nrounds = 1000, eta = c(0.01,0.05,0.1), max_depth = c(2,4,6,8,10,14), gamma=1, min_child_weight = 7, subsample = 0.8, colsample_bytree = 0.8)

  xgb_model <-train(formula, data=data, method="xgbTree", trControl=myControl,tuneGrid=xgb.grid,verbose=T, metric="Kappa",nthread =1)

  svmLinear_model <-train(formula, data=data, method="svmLinear", trControl=myControl)

  svmRadial_model <-train(formula, data=data, method="svmRadial", trControl=myControl)

  svmPoly_model <-train(formula, data=data, method="svmPoly", trControl=myControl)

  knn_model <-train(formula, data=data, method="knn", trControl=myControl)

  models <- list(svmPoly=svmPoly_model, rf = rf_model, glm = glm_model, glmnet=glmnet_model, xgboost=xgb_model, svmLinear=svmLinear_model, svmRadial=svmRadial_model, knearest=knn_model)

  resampled <- resamples(models)

  return(resampled)

}
# Run 1 with (most) available variables

train.mod=data[1:train.stop,]

formula=as.formula(Survived~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked)

resampled=suppressWarnings(compareML(train.mod, formula))

dotplot(resampled, metric='Accuracy')

top.run1=max(summary(resampled)$statistics$Accuracy[,'Median'])

print(top.run1)
# Run 2 with some feature engineering

data$title=gsub('(\\..*)|(.*, )', '', data$Name) # Title variable

tmp=count(as.character(data$title))

rare=tmp[tmp$freq<10,1]

data[data$title %in% rare, 'title']='rare'

data$title=as.factor(data$title) 

data$deck=as.factor(sapply(as.character(data$Cabin), function(x) strsplit(x,NULL)[[1]][1])) # Deck variable

data[is.na(data[,'deck']),'deck']=names(which.max(table(na.omit(data[,'deck']))))

data$Ticket2=toupper(gsub('( )|([.])|(/)', '', gsub("[[:digit:]]", '', data$Ticket)))

tmp=count(as.character(data$Ticket2))

rare=tmp[tmp$freq<10,1]

data[data$Ticket2 %in% rare, 'Ticket2']='rare'

data$Ticket2=as.factor(data$Ticket2) 

train.mod=data[1:train.stop,]

formula=as.formula(Survived~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked+title+deck+Ticket2)

train.mod=data[1:train.stop,]

resampled=suppressWarnings(compareML(train.mod, formula))

dotplot(resampled, metric='Accuracy')

top.run2=max(summary(resampled)$statistics$Accuracy[,'Median'])

print(top.run2)
# Run 3 - mice imputation

data=bind_rows(train, test)

data$Survived=as.factor(as.character(data$Survived))

data$Pclass=as.factor(as.character(data$Pclass))

data$Embarked=as.factor(as.character(data$Embarked))

data$title=gsub('(\\..*)|(.*, )', '', data$Name) # Title variable

tmp=count(as.character(data$title))

rare=tmp[tmp$freq<10,1]

data[data$title %in% rare, 'title']='rare'

data$title=as.factor(data$title) 

data$deck=as.factor(sapply(as.character(data$Cabin), function(x) strsplit(x,NULL)[[1]][1])) # Deck variable

data.4mice=data[c("Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","title", "deck")]

data.mice=complete(mice(data.4mice))

data.imp=cbind(data[,c("PassengerId", "Survived","Name", "Ticket","Cabin")], data.mice)

#split

train=data.imp[1:train.stop,]

formula=as.formula(Survived~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked+title+deck)

resampled=surpressWarnings(compareML(train, formula))

dotplot(resampled, metric='Accuracy')

top.run4=max(summary(resampled)$statistics$Accuracy[,'Median'])

print(top.run4)