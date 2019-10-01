# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function

library(data.table)

library(caret)



BinByCount <- function(x, n=10) {

	findInterval(x, sort(quantile(x, (1:n) * (1/n) - (1/n), na.rm=TRUE)))

}



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



# Any results you write to the current directory are saved as output.
all_train <- fread('../input/train.csv')

test  <- fread('../input/test.csv')

summary(all_train)



# combine them as a whole

test$Survived <- NA

in_train <- createDataPartition(all_train$Survived, p = 7/8, list = FALSE)

train = all_train[in_train]

pre_test = all_train[-in_train]



nrow(train)

nrow(pre_test)

train[, Category := 'train']

pre_test[, Category := 'cv']

test[, Category := 'test']

full <- rbind(train, pre_test, test)

# train[, list(Cabin)]

train[, NewCabin := sapply(Cabin, function(x) {length(strsplit(x, ' ')[[1]])})]

full[, NewCabin := sapply(Cabin, function(x) {length(strsplit(x, ' ')[[1]])})]

table(full$NewCabin)
columns = colnames(train)

par(mfrow=c(3, 3))

for (column in c('Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Fare', 'NewCabin', 'Embarked')) {

    if (typeof(train[, eval(as.name(column))][1]) == 'double') {

        train[, Regressor := BinByCount(train[, eval(as.name(column))])]

    } else {

        train[, Regressor := as.factor(train[, eval(as.name(column))])]

    }

    train_summary = train[, list(FirstQuart=quantile(Survived, 0.25), Mean=mean(Survived), ThirdQuart=quantile(Survived, 0.75)), by=Regressor]

    plot(train_summary$Regressor, train_summary$Mean)

    title(column)

}
columns = colnames(train)

par(mfrow=c(3, 3))

for (column in c('Pclass', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked')) {

    if (typeof(train[, eval(as.name(column))][1]) == 'double') {

        plot(train[, eval(as.name(column))], train$Age)

    } else {

        plot(as.factor(train[, eval(as.name(column))]), train$Age, )

    }

    title(column)

}
full[, NewAge := Age]

set.seed(123)

full[is.na(Age), NewAge := sample(na.omit(full$Age), sum(is.na(full$Age)))]

par(mfrow=c(2,1))

hist(full$Age, freq=F)

hist(full$NewAge, freq=F)
full[, Title := sapply(Name, function(x) {strsplit(strsplit(x,", ")[[1]][2], '\\.')[[1]][1]})]

full[, LastName := sapply(Name, function(x) {strsplit(x,", ")[[1]][2]})]

table(full$Title)

full[!(Title %in% c('Master', 'Miss', 'Mr', 'Mrs')), Title := 'Other']

full[, Title := as.factor(Title)]

table(full$Title)
full[, Survived := as.factor(Survived)]

ctrl <- trainControl(method = "repeatedcv", repeats = 5, classProbs = TRUE)

rpart_mdl <- train(Survived ~ Pclass + Sex + NewAge + Title + SibSp + Parch + NewCabin + 

                   Fare + Embarked, data=full[Category == 'train'],

                   method = "rpart")

rpart_mdl

confusionMatrix(rpart_mdl)

full[Category %in% c('train', 'cv'), RpartSurvived := predict(rpart_mdl, full[Category %in% c('train', 'cv')])]

full[Category == 'cv', sum(RpartSurvived == Survived)/length(Survived)]
rpart_mdl2 <- train(Survived ~ Pclass + Sex + Age + Title + SibSp + Parch + NewCabin + 

                   Fare + Embarked, data=full[Category == 'train'],

                   method = "rpart",

                   preProc = c("center", "scale", "knnImpute"),

                   na.action = na.pass

                   )

rpart_mdl2

confusionMatrix(rpart_mdl2)

full[Category %in% c('train', 'cv'), RpartSurvived2 := predict(rpart_mdl2, full[Category %in% c('train', 'cv')], na.action=na.pass)]

full[Category == 'cv', sum(RpartSurvived2 == Survived)/length(Survived)]

# almost same as random age assignment, maybe slightly better looking at accuracy
rf_mdl <- train(Survived ~ Pclass + Sex + Age + Title + SibSp + Parch + NewCabin + 

                   Fare + Embarked, data=full[Category == 'train'],

                   method = "rf",

                   preProc = c("center", "scale", "knnImpute"),

                   na.action = na.pass

                   )

rf_mdl

confusionMatrix(rf_mdl)

full[Category %in% c('train', 'cv'), RfSurvived := predict(rf_mdl, full[Category %in% c('train', 'cv')], na.action=na.pass)]

full[Category == 'cv', sum(RfSurvived == Survived)/length(Survived)]
glm_mdl <- train(Survived ~ Pclass + Sex + Age + Title + SibSp + Parch + NewCabin + 

                   Fare + Embarked, data=full[Category == 'train'],

                   method = "glm",

                   preProc = c("center", "scale", "knnImpute"),

                   na.action = na.pass

                   )

glm_mdl

confusionMatrix(glm_mdl)

full[Category %in% c('train', 'cv'), GlmSurvived := predict(glm_mdl, full[Category  %in% c('train', 'cv')], na.action=na.pass)]

full[Category == 'cv', sum(GlmSurvived == Survived)/length(Survived)]
svm_mdl <- train(Survived ~ Pclass + Sex + Age + Title + SibSp + Parch + NewCabin + 

                   Fare + Embarked, data=full[Category == 'train'],

                   method = "svmLinearWeights2",

                   preProc = c("center", "scale", "knnImpute"),

                   na.action = na.pass

                   )

svm_mdl

confusionMatrix(svm_mdl)

full[Category %in% c('train', 'cv'), SvmSurvived := predict(svm_mdl, full[Category  %in% c('train', 'cv')], na.action=na.pass)]

full[Category == 'cv', sum(SvmSurvived == Survived)/length(Survived)]
full[Category %in% c('cv', 'train'), sum(RpartSurvived == Survived)/length(Survived)]
full[Category %in% c('cv', 'train'), list(PctRpart=sum(RpartSurvived == Survived)/length(Survived), 

                                          PctRpart2=sum(RpartSurvived2 == Survived)/length(Survived), 

                                          PctRf=sum(RfSurvived == Survived)/length(Survived), 

                                          PctGlm=sum(GlmSurvived == Survived)/length(Survived), 

                                          PctSvm=sum(SvmSurvived == Survived)/length(Survived))]