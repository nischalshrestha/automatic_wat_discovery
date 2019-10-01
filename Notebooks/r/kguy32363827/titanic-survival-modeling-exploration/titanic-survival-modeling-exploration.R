# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



# Any results you write to the current directory are saved as output.
set.seed(1293847)

trn_raw = read.csv('../input/train.csv')

HO_idx <- sample(seq_len(nrow(trn_raw)), size = floor(0.2 * nrow(trn_raw)))

trn_r = trn_raw[-HO_idx,]

val_r = trn_raw[HO_idx,]

tst_r = read.csv('../input/test.csv')

str(trn_r)

str(val_r)

str(tst_r)
trn_m = trn_r

val_m = val_r

tst_m = tst_r
rbind(trn_m[trn_m$Embarked == "",],val_m[val_m$Embarked == "",])
table(trn_m$Embarked)/sum(table(trn_m$Embarked))
trn_m[trn_m$Embarked == "",c('Embarked')] = 'S'

val_m[val_m$Embarked == "",c('Embarked')] = 'S'
colSums(is.na(trn_m))

colSums(is.na(val_m))

colSums(is.na(tst_m))
# Technique picked up from here: https://www.r-bloggers.com/imputing-missing-data-with-r-mice-package/:

pMiss <- function(x){sum(is.na(x))/length(x)*100}

apply(trn_m,2,pMiss)[apply(trn_m,2,pMiss)>0]

apply(val_m,2,pMiss)[apply(val_m,2,pMiss)>0]

apply(tst_m,2,pMiss)[apply(tst_m,2,pMiss)>0]
sd(trn_m$Fare)

median(trn_m$Fare)

fill_Fare = mean(trn_m$Fare)

tst_m$Fare_orig = tst_m$Fare
tst_m.origfare_noNA = tst_m[complete.cases(tst_m[,c('Fare')]),c('Fare')]

mean(tst_m.origfare_noNA)

sd(tst_m.origfare_noNA)



tst_m[is.na(tst_m$Fare),c('Fare')] = fill_Fare

mean(tst_m$Fare)

sd(tst_m$Fare)
tst_m = subset(tst_m, select=-c(Fare_orig))
fill_Age_mean = mean(trn_m[complete.cases(trn_m[,c('Age')]),c('Age')])



fill_Age_mean

sd(trn_m[complete.cases(trn_m[,c('Age')]),c('Age')])

median(trn_m[complete.cases(trn_m[,c('Age')]),c('Age')])
trn_m$Age_comp_FM = trn_m$Age

val_m$Age_comp_FM = val_m$Age

tst_m$Age_comp_FM = tst_m$Age
trn_m[is.na(trn_m$Age_comp_FM),c('Age_comp_FM')] = fill_Age_mean

val_m[is.na(val_m$Age_comp_FM),c('Age_comp_FM')] = fill_Age_mean

tst_m[is.na(tst_m$Age_comp_FM),c('Age_comp_FM')] = fill_Age_mean



mean(trn_m$Age_comp_FM)

sd(trn_m$Age_comp_FM)

median(trn_m$Age_comp_FM)
par(mfrow=c(1,2))

hist(trn_m$Age, freq = FALSE, main = 'Age: Original', ylim = c(0,0.045))

hist(trn_m$Age_comp_FM, freq = FALSE, main = 'Age: Filled with Mean', ylim = c(0,0.045))
library(mice)

cols_for_age_imp = c("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked")

trn_imp_ds = trn_m[names(trn_m) %in% cols_for_age_imp]

trn_imp <- complete(mice(trn_imp_ds, seed = 123432))

trn_m$Age_comp_MICE = trn_imp$Age



val_imp_ds = val_m[names(val_m) %in% cols_for_age_imp]

val_imp <- complete(mice(val_imp_ds, seed = 123412348))

val_m$Age_comp_MICE = val_imp$Age



tst_imp_ds = tst_m[names(tst_m) %in% cols_for_age_imp]

tst_imp <- complete(mice(tst_imp_ds, seed = 123412348))

tst_m$Age_comp_MICE = tst_imp$Age
par(mfrow=c(3,3))

hist(trn_m$Age, freq= FALSE, main = "Training Data:  Original Age", ylim = c(0,0.045))

hist(trn_m$Age_comp_FM, freq= FALSE, main = "Training Data: Age via Mean", ylim = c(0,0.045))

hist(trn_m$Age_comp_MICE, freq= FALSE, main = "Training Data: Age via MICE", ylim = c(0,0.045))



hist(val_m$Age, freq= FALSE, main = "Vdalidation Data:  Original Age", ylim = c(0,0.045))

hist(val_m$Age_comp_FM, freq= FALSE, main = "Validation Data: Age via Mean", ylim = c(0,0.045))

hist(val_m$Age_comp_MICE, freq= FALSE, main = "Validation Data: Age via MICE", ylim = c(0,0.045))



hist(tst_m$Age, freq= FALSE, main = "Test Data:  Original Age", ylim = c(0,0.045))

hist(tst_m$Age_comp_FM, freq= FALSE, main = "Test Data: Age via Mean", ylim = c(0,0.045))

hist(tst_m$Age_comp_MICE, freq= FALSE, main = "Test Data: Age via MICE", ylim = c(0,0.045))
age_compare = data.frame( Dataset =character(), Calculation = character(),

                   Mean = double(), StD = double(), 

                   stringsAsFactors=FALSE)



age_compare[1,] = c('Train', 'Original', mean(trn_m[complete.cases(trn_m[,c('Age')]),c('Age')]), sd(trn_m[complete.cases(trn_m[,c('Age')]),c('Age')]))

age_compare[2,] = c('Train', 'Filled by Mean', mean(trn_m$Age_comp_FM), sd(trn_m$Age_comp_FM))

age_compare[3,] = c('Train', 'Filled by MICE', mean(trn_m$Age_comp_MICE), sd(trn_m$Age_comp_MICE))



age_compare[4,] = c('Val', 'Original', mean(val_m[complete.cases(val_m[,c('Age')]),c('Age')]), sd(val_m[complete.cases(val_m[,c('Age')]),c('Age')]))

age_compare[5,] = c('Val', 'Filled by Mean', mean(val_m$Age_comp_FM), sd(val_m$Age_comp_FM))

age_compare[6,] = c('Val', 'Filled by MICE', mean(val_m$Age_comp_MICE), sd(val_m$Age_comp_MICE))



age_compare[7,] = c('Test', 'Original', mean(tst_m[complete.cases(tst_m[,c('Age')]),c('Age')]), sd(tst_m[complete.cases(tst_m[,c('Age')]),c('Age')]))

age_compare[8,] = c('Test', 'Filled by Mean', mean(tst_m$Age_comp_FM), sd(tst_m$Age_comp_FM))

age_compare[9,] = c('Test', 'Filled by MICE', mean(tst_m$Age_comp_MICE), sd(tst_m$Age_comp_MICE))



age_compare
bp <- barplot(table(trn_m$Survived), main = 'Survival')

text(bp,table(trn_m$Survived)*.9,labels=table(trn_m$Survived))

text(bp,20,labels=round((table(trn_m$Survived)/sum(table(trn_m$Survived)))*100,2))

trn_m$Survived_F = factor(trn_m$Survived)

val_m$Survived_F = factor(val_m$Survived)
trn_m$Pclass = factor(trn_m$Pclass)

trn_m$Embarked = factor(trn_m$Embarked)

trn_m$Sex = factor(trn_m$Sex)



val_m$Pclass = factor(val_m$Pclass)

val_m$Embarked = factor(val_m$Embarked)

val_m$Sex = factor(val_m$Sex)



tst_m$Pclass = factor(tst_m$Pclass)

tst_m$Embarked = factor(tst_m$Embarked)

tst_m$Sex = factor(tst_m$Sex)





table(trn_m$Pclass, trn_m$Survived)

print('Class')



table(trn_m$Embarked,trn_m$Survived)

print('Embarked')



table(trn_m$Sex,trn_m$Survived)

print('Sex')
par(mfrow=c(2,2))

boxplot(Fare~Survived, data=trn_m, main = "Fare")

boxplot(Age_comp_MICE~Survived, data=trn_m, main = "Age (filled via MICE)")

boxplot(SibSp~Survived, data=trn_m, main = "Siblings/Spouses")

boxplot(Parch~Survived, data=trn_m, main = "Parents/Children")
trn_m$HasCabin <- factor(ifelse(trn_m$Cabin == "", c(0), c(1)))

trn_m$FamilySize = trn_m$SibSp + trn_m$Parch +1

trn_m$HasFamily <- factor(ifelse(trn_m$FamilySize == 1, c(0), c(1)))



val_m$HasCabin <- factor(ifelse(val_m$Cabin == "", c(0), c(1)))

val_m$FamilySize = val_m$SibSp + val_m$Parch +1

val_m$HasFamily <- factor(ifelse(val_m$FamilySize == 1, c(0), c(1)))



tst_m$HasCabin <- factor(ifelse(tst_m$Cabin == "", c(0), c(1)))

tst_m$FamilySize = tst_m$SibSp + tst_m$Parch +1

tst_m$HasFamily <- factor(ifelse(tst_m$FamilySize == 1, c(0), c(1)))
table(trn_m$HasCabin,trn_m$Survived)

print('HasCabin')



table(trn_m$HasFamily,trn_m$Survived)

print('HasFamily')
table(trn_m[,c("Pclass", "HasCabin")])

for (i in 1:3){

  cat('Pclass', i,': ', ((table(trn_m[,c("Pclass", "HasCabin")])[i,]/sum(table(trn_m[,c("Pclass", "HasCabin")])[i,]))[2])*100, '%\n')

}
trn_m$NonFC_wCabin <- ifelse(trn_m$Pclass != 1 & trn_m$HasCabin == 1, c(1), c(0)) 

table(trn_m[,c("NonFC_wCabin", "Survived")])

for (i in 1:2){

  cat('Survival rate of Non-First Class + Cabin Satus of ', i-1, ': ', ((table(trn_m[,c("NonFC_wCabin", "Survived")])[i,]/sum(table(trn_m[,c("NonFC_wCabin", "Survived")])[i,]))[2])*100, '%\n')

}
cat(dim(trn_m[trn_m$Pclass == 1 & trn_m$Cabin == "" & trn_m$SibSp==0 & trn_m$Parch==0,])[1], ' people traveling alone; (', 

    (dim(trn_m[trn_m$Pclass == 1 & trn_m$Cabin == "" & trn_m$SibSp==0 & trn_m$Parch==0,])[1]/table(trn_m[,c("Pclass", "HasCabin")])[1,1])*100, '% of those in PClass 1 without cabins)')
cabintable = table(trn_m$Cabin)

cab_occupancy = data.frame(cabintable)

colnames(cab_occupancy) = c('Cabin', 'Cab_Occ')

trn_m = merge(trn_m,cab_occupancy, by = 'Cabin')



trn_m$SharedCabin = ifelse(trn_m$Cab_Occ >1, c(1), c(0)) 

cat(table(trn_m$FamilySize,trn_m$SharedCabin)[1,2], 'people traveling without a parental, spousal, sibling or child relationship were sharing cabins.')
trn_m$Age_Bin = cut(trn_m$Age_comp_MICE, breaks = c(0, 1, 12, 18, 60, 200))

val_m$Age_Bin = cut(val_m$Age_comp_MICE, breaks = c(0, 1, 12, 18, 60, 200))

tst_m$Age_Bin = cut(tst_m$Age_comp_MICE, breaks = c(0, 1, 12, 18, 60, 200))

table(trn_m$Age_Bin, trn_m$Survived)
boxplot(Fare~Pclass,data=trn_m, main="Fare vs Class", 

  	xlab="pClass", ylab="Fare")
## Create an empty dataframe which will house the Chi-Squared results

Chi = data.frame( var_name =character(), val = character(),

                   No_Survival = double(), Survived= double(), 

                   pvalue= double(),

                   stringsAsFactors=FALSE)

f=0

chirow = 0

disc_var = c("Pclass", "Sex", "Embarked", "HasCabin", "HasFamily", "Age_Bin")

cat('Running chi-squared tests on discrete independent variables...')

for (i in 1:length(disc_var)){

##Run the chi2 test on the next dependent variable

  tbl <- table(trn_m[,c(disc_var[i])],trn_m$Survived_F)

  t=chisq.test(tbl)

  ## grab the test results to populate the next two rows of the Chi dataframe (one row for each value of the independent variable)

  for (f in 1:dim(tbl)[1]){

    chirow = chirow+1

    Chi[chirow, c(1,2)] <- c(disc_var[i], rownames(tbl)[f])

    Chi[chirow, c(3,4,5)]<-c(tbl[f,1], tbl[f,2],t$p.value)}

}

Chi = Chi[with(Chi, order(pvalue,var_name,val)), ]

Chi
train = trn_m[,append(disc_var,c("Survived_F"))]

val = val_m[,append(disc_var,c("Survived_F"))]

test = tst_m[,disc_var]

library(ROSE)

trn_cnt = as.numeric(table(train$Survived_F)[1]*2)

train_os <- ovun.sample(Survived_F ~ ., data = train, method = "over",N = trn_cnt)$data

table(train_os$Survived_F)
# creating training variables & empty dataframe to store model assessment results

train_x = train[,-which(names(train) %in% c("Survived_F"))]

train_os_x = train_os[,-which(names(train_os) %in% c("Survived_F"))]

val_x = val[,-which(names(val) %in% c("Survived_F"))]



Model_compare = data.frame( Model =character(), Dataset = character(),

                   AUC = double(), Accuracy= double(), 

                   Sensitivity= double(), Specificity = double(), Precision = double(), 

                   stringsAsFactors=FALSE)
# Naive Bayes

set.seed(12762)



library(e1071)

library(caret)

library(pROC)

library(ROCR)

model_NB_ubal = naiveBayes(Survived_F~., data = train)

p_NB_ubal = predict(model_NB_ubal,val, type="class")

roc_NB_ubal <- roc(val_m$Survived, as.numeric(p_NB_ubal)-1)

cm_NB_ubal <- confusionMatrix(data=p_NB_ubal,reference=val$Survived_F, positive= '1')



Model_compare[1,] = c('Naive Bayes', 'Unbalanced', auc(roc_NB_ubal), 

                     as.numeric(cm_NB_ubal$overall[1]), # Accuracy

                     as.numeric(cm_NB_ubal$byClass[1]), # Sensitivity

                     as.numeric(cm_NB_ubal$byClass[2]), # Specificity

                     as.numeric(cm_NB_ubal$byClass[5])) # Precision





model_NB_bal = naiveBayes(Survived_F~., data = train_os)

p_NB_bal = predict(model_NB_bal,val, type="class")

roc_NB_bal <- roc(val_m$Survived, as.numeric(p_NB_bal)-1)

cm_NB_bal <- confusionMatrix(data=p_NB_bal,reference=val$Survived_F, positive= '1')



Model_compare[2,] = c('Naive Bayes', 'Balanced', auc(roc_NB_bal), 

                     as.numeric(cm_NB_bal$overall[1]), # Accuracy

                     as.numeric(cm_NB_bal$byClass[1]), # Sensitivity

                     as.numeric(cm_NB_bal$byClass[2]), # Specificity

                     as.numeric(cm_NB_bal$byClass[5])) # Precision

# Logistic Regression

set.seed(23489234)

model_LR_ubal <- glm(Survived_F ~.,family=binomial(link='logit'),   data=train)

p_LR_ubal = predict(model_LR_ubal,val, type="response")

p_LR_ubal <- ifelse(p_LR_ubal > 0.5,1,0)

roc_LR_ubal <- roc(val_m$Survived, as.numeric(p_LR_ubal)-1)

cm_LR_ubal <- confusionMatrix(data=p_LR_ubal,reference=val$Survived_F, positive= '1')



Model_compare[3,] = c('Logistic Regression', 'Unbalanced', auc(roc_LR_ubal), 

                     as.numeric(cm_LR_ubal$overall[1]), # Accuracy

                     as.numeric(cm_LR_ubal$byClass[1]), # Sensitivity

                     as.numeric(cm_LR_ubal$byClass[2]), # Specificity

                     as.numeric(cm_LR_ubal$byClass[5])) # Precision



model_LR_bal <- glm(Survived_F ~.,family=binomial(link='logit'),   data=train_os)

p_LR_bal = predict(model_LR_bal,val, type="response")

p_LR_bal <- ifelse(p_LR_bal > 0.5,1,0)

roc_LR_bal <- roc(val_m$Survived, as.numeric(p_LR_bal)-1)

cm_LR_bal <- confusionMatrix(data=p_LR_bal,reference=val$Survived_F, positive= '1')



Model_compare[4,] = c('Logistic Regression', 'Balanced', auc(roc_LR_bal), 

                     as.numeric(cm_LR_bal$overall[1]), # Accuracy

                     as.numeric(cm_LR_bal$byClass[1]), # Sensitivity

                     as.numeric(cm_LR_bal$byClass[2]), # Specificity

                     as.numeric(cm_LR_bal$byClass[5])) # Precision
## Random forest: initial comparison of balanced vs. imbalanced data

## Resource: https://www.r-bloggers.com/random-forests-in-r/

library(randomForest)

set.seed(10981)

model_RF_ubal = randomForest(Survived_F ~., data = train)

model_RF_bal = randomForest(Survived_F ~., data = train_os)

par(mfrow=c(1,2))

plot(model_RF_ubal, main = 'Unbalanced', ylim = c(0,.4))

plot(model_RF_bal, main = 'Balanced \n(via oversampling)', ylim = c(0,.4))
model_RF_bal
rf_err = data.frame(model_RF_bal$err.rate)

rf_err$combined_err = rowSums(rf_err)

rf_err[which.min(rf_err$combined_err),]
# calculating optimal number of variables per split

set.seed(1239487)

ntree = 116

best_mtry <-tuneRF(x=train_os_x, y=train_os$Survived_F, ntreeTry=ntree, stepFactor=1.5, improve=0.01, trace=TRUE, plot=FALSE, dobest=TRUE)



mtry = best_mtry[which(best_mtry == min(best_mtry), arr.ind=TRUE)[1],1]

cat('Optimal Number of Trees: ', ntree)

cat('\nOptimal Number of Variables sampled per split: ', mtry)
# Random Forest Model

set.seed(23489234)

model_RF_ubal = randomForest(Survived_F ~., data = train, mtry=mtry, ntree = ntree)

p_RF_ubal = predict(model_RF_ubal,val, type="response")

roc_RF_ubal <- roc(val_m$Survived, as.numeric(p_RF_ubal)-1)

cm_RF_ubal <- confusionMatrix(data=p_RF_ubal,reference=val$Survived_F, positive= '1')



Model_compare[5,] = c('Random Forest', 'Unbalanced', auc(roc_RF_ubal), 

                     as.numeric(cm_RF_ubal$overall[1]), # Accuracy

                     as.numeric(cm_RF_ubal$byClass[1]), # Sensitivity

                     as.numeric(cm_RF_ubal$byClass[2]), # Specificity

                     as.numeric(cm_RF_ubal$byClass[5])) # Precision



model_RF_bal = randomForest(Survived_F ~., data = train_os, mtry=mtry, ntree = ntree)

p_RF_bal = predict(model_RF_bal,val, type="response")

roc_RF_bal <- roc(val_m$Survived, as.numeric(p_RF_bal)-1)

cm_RF_bal <- confusionMatrix(data=p_RF_bal,reference=val$Survived_F, positive= '1')



Model_compare[6,] = c('Random Forest', 'Balanced', auc(roc_RF_bal), 

                     as.numeric(cm_RF_bal$overall[1]), # Accuracy

                     as.numeric(cm_RF_bal$byClass[1]), # Sensitivity

                     as.numeric(cm_RF_bal$byClass[2]), # Specificity

                     as.numeric(cm_RF_bal$byClass[5])) # Precision
Model_compare
importance    <- importance(model_RF_bal)

varImportance <- data.frame(Variables = row.names(importance), 

                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

plot(varImportance)
final_predict_prob = predict(model_LR_bal,test, type="response")

final_prediction = ifelse(final_predict_prob > 0.5,1,0)

solution_file = data.frame("PassengerId" = tst_m$PassengerId, "Survived" = final_prediction)

#write.csv(solution_file, file = 'titanic_survival_prediction__oversampledLR.csv', row.names = FALSE)
LR = final_prediction

ensemble_predict = as.data.frame(LR)

ensemble_predict$NB = as.numeric(predict(model_NB_bal,test, type="class"))-1

ensemble_predict$RF = as.numeric(predict(model_RF_bal,test, type="response"))-1

ensemble_predict$LR = as.numeric(ensemble_predict$LR)

ensemble_predict$NB = as.numeric(ensemble_predict$NB)

ensemble_predict$RF = as.numeric(ensemble_predict$RF)

ensembled = round(rowMeans(ensemble_predict),0)

#write.csv(ensembled, file = 'titanic_survival_prediction__simpleEnsemble.csv', row.names = FALSE)
sum(ensembled == final_prediction)/length(ensembled)*100