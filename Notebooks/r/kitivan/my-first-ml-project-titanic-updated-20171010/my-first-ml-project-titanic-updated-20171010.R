#load the library

library(plyr)

library(dplyr)

library(ggplot2)

library(caret)

library(gridExtra)

library(modeest)

library(randomForest)

library(fastAdaboost)

library(kernlab)

library(xgboost)

#File input

trainset <- read.csv('../input/train.csv', na.strings=c("", "NA"), sep=",", header=TRUE, stringsAsFactors = FALSE)

testset  <- read.csv('../input/test.csv',  na.strings=c("", "NA"), sep=",", header=TRUE, stringsAsFactors = FALSE)
str(trainset)

str(testset)



head(trainset)
sapply(trainset, function (x) sum(is.na(x)))

sapply(testset, function (x) sum(is.na(x)))
#check for the passenger Pclass and location of embarkation

testset[which(is.na(testset$Fare)),]



#Combine dataset with column Pclass, Embarked and Fare

Fare_impute_df <- data.frame(Pclass = c(trainset$Pclass, testset$Pclass), Embarked = c(trainset$Embarked, testset$Embarked), Fare = c(trainset$Fare, testset$Fare))

Fare_impute_df %>%

  filter (Pclass == 3 & Embarked == "S") %>%

  ggplot (aes(x=Fare)) +

  geom_histogram()
testset$Fare[153] <- median(Fare_impute_df$Fare[Fare_impute_df$Pclass == 3 & Fare_impute_df$Embarked == "S"], na.rm=TRUE)
trainset[which(is.na(trainset$Embarked)),]



embarked_impute_df <- data.frame(Pclass = c(trainset$Pclass, testset$Pclass), Embarked = as.factor(c(trainset$Embarked, testset$Embarked)), Fare = c(trainset$Fare, testset$Fare))
embarked_impute_df %>%

  na.omit() %>%

  filter (Pclass == 1 & Fare >= 75 & Fare <= 85) %>%

  group_by(Embarked) %>%

  summarise(count = n())
trainset$Embarked[c(62,830)] = "C"
Age_impute <- data.frame(Survived = trainset$Survived, Age = trainset$Age)

Age_impute$Agegroup1<-cut(Age_impute$Age, seq(0,80,20), labels=c(1:4))

Age_impute$Agegroup2<-cut(Age_impute$Age, seq(0,80,10), labels=c(1:8))

Age_impute$Agegroup3<-cut(Age_impute$Age, seq(0,80,5), labels=c(1:16))



#Plot the graph between survival and age group

Agegroup1 <- ggplot(Age_impute, aes(x=factor(Agegroup1), fill=factor(Survived))) +

  geom_bar(position = "fill")



Agegroup2 <- ggplot(Age_impute, aes(x=factor(Agegroup2), fill=factor(Survived))) +

  geom_bar(position = "fill")



Agegroup3 <- ggplot(Age_impute, aes(x=factor(Agegroup3), fill=factor(Survived))) +

  geom_bar(position = "fill")



Agegroup1

Agegroup2

Agegroup3
Age_impute<-Age_impute %>%

  mutate(Agegroup4 = ifelse(Age<=15,1,

                            ifelse(15<Age & Age<=60,2,

                                   ifelse(Age>60,3, "NA"))))



Agegroup4 <- ggplot(Age_impute, aes(x=factor(Agegroup4), fill=factor(Survived))) +

  geom_bar(position = "fill")



#COmbine 4 graph together on one page

grid.arrange(Agegroup1, Agegroup2, Agegroup3, Agegroup4, nrow=2, ncol=2)
chisq.test(table(Age_impute$Survived, Age_impute$Agegroup1, useNA = "ifany"))

chisq.test(table(Age_impute$Survived, Age_impute$Agegroup2, useNA = "ifany"))

chisq.test(table(Age_impute$Survived, Age_impute$Agegroup3, useNA = "ifany"))

chisq.test(table(Age_impute$Survived, Age_impute$Agegroup4, useNA = "ifany"))
trainset<-trainset %>%

  mutate(Agegroup = ifelse(Age<=15,"kids",ifelse(15<Age & Age<=60,"adults",ifelse(Age>60,"elderly", "NA")))) 

testset<-testset %>%

  mutate(Agegroup = ifelse(Age<=15,"kids",ifelse(15<Age & Age<=60,"adults",ifelse(Age>60,"elderly", "NA")))) 
trainset$Cabin <- NULL

testset$Cabin <- NULL
#Response variable distribution. 

#Although it is not 50:50 balance, it is not too biased therefore no oversampling needed.

ggplot(trainset, aes(x=Survived)) +

  geom_bar()
#Distribution of response variable conditional to Pclass. 

#Pclass 1 has highest survival rate and Pclass 3 has lowest survival rate.

ggplot(trainset, aes(x=as.factor(Survived), fill=as.factor(Pclass))) +

  geom_bar(position="fill")
#Distribution of response variable conditional to Sex

#Female has higher survival rate then Male. 

ggplot(trainset, aes(x=as.factor(Survived), fill=as.factor(Sex))) +

  geom_bar(position="fill")
#Distribution of age group conditional to response variable

#We have investigated it in above imputation.

ggplot(trainset, aes(x=as.factor(Agegroup), fill=as.factor(Survived))) +

  geom_bar(position="fill")
#Distribution of Fare conditional to response variable

#Higher the fare price, higher survival rate.

summary(trainset$Fare)

ggplot(trainset, aes(fill=as.factor(Survived), x=Fare)) +

  geom_histogram(position = "dodge")
#Distribution of Embarked conditional to response variable

#It is interesting that there are different survival rates according to port of Embarkation.

ggplot(trainset, aes(x=as.factor(Embarked), fill=as.factor(Survived))) +

  geom_bar(position="fill")
#Distribution of SibSp & Parch conditional to response variable

#We could see the passenger with no SibSP or Parch has less survival rate.

ggplot(trainset, aes(fill=as.factor(Survived), x=SibSp)) +

  geom_histogram(position = "dodge", binwidth = 1)



ggplot(trainset, aes(fill=as.factor(Survived), x=Parch)) +

  geom_histogram(position = "dodge", binwidth = 1)
#Create a new column company. 0 is alone, 1 is with company

trainset<-trainset %>%

  mutate(company = ifelse(SibSp+Parch>=1,1,0))

ggplot(trainset, aes(fill=as.factor(Survived), x=as.factor(company))) +

  geom_bar(position = "fill")
#Update the testset with the same transformation

testset<-testset %>%

  mutate(company = ifelse(SibSp+Parch>=1,1,0))
#Convert the categorical variable to Factor before training.

trainset$Survived <- factor(trainset$Survived)

trainset$Pclass <- factor(trainset$Pclass)

trainset$Sex <- factor(trainset$Sex)

trainset$Embarked <- factor(trainset$Embarked)

trainset$Agegroup <- factor(trainset$Embarked, exclude=NULL)

trainset$company <- factor(trainset$company)



#Modelling

set.seed(42)

#rf_model <- train(Survived ~ Pclass + Sex + Fare + Embarked + Agegroup + company,

#                  data=trainset,

#                  method = "rf",

#                  trControl=trainControl(method="cv", number=5))





#glm_model <- train(Survived ~ Pclass + Sex + Fare + Embarked + Agegroup + company,

#                  data=trainset,

#                  method = "glmnet",

#                  trControl=trainControl(method="cv", number=5))



#adaboost_model <- train(Survived ~ Pclass + Sex + Fare + Embarked + Agegroup + company,

#                  data=trainset,

#                  method = "adaboost",

#                  trControl=trainControl(method="cv", number=5))



#svmLinear_model <- train(Survived ~ Pclass + Sex + Fare + Embarked + Agegroup + company,

#                  data=trainset,

#                  method = "svmLinear",

#                  trControl=trainControl(method="cv", number=5))



#xgbTree_model <- train(Survived ~ Pclass + Sex + Fare + Embarked + Agegroup + company,

#                  data=trainset,

#                  method = "xgbTree",

#                  trControl=trainControl(method="cv", number=5))



#rf_perdict <- predict(rf_model, data = trainset)

#glm_perdict <- predict(glm_model, data = trainset)

#adaboost_perdict <- predict(adaboost_model, data = trainset)

#svmLinear_perdict <- predict(svmLinear_model, data = trainset)

#xgbTree_perdict <- predict(xgbTree_model, data = trainset)



#table(trainset$Survived, rf_perdict)

#table(trainset$Survived, glm_perdict)

#table(trainset$Survived, adaboost_perdict)

#table(trainset$Survived, svmLinear_perdict)

#table(trainset$Survived, xgbTree_perdict)
#Since it takes long time to run above script here, i copy the results from my notebook.



#table(trainset$Survived, rf_perdict)

#   rf_perdict

#      0   1

#  0 528  21

#  1  56 286

#table(trainset$Survived, rf_perdict)

#   rf_perdict

#      0   1

#  0 528  21

#  1  56 286

#table(trainset$Survived, glm_perdict)

#   glm_perdict

#      0   1

#  0 468  81

#  1 109 233

#table(trainset$Survived, adaboost_perdict)

#   adaboost_perdict

#      0   1

#  0 523  26

#  1  51 291

#table(trainset$Survived, svmLinear_perdict)

#   svmLinear_perdict

#      0   1

#  0 468  81

#  1 109 233

#table(trainset$Survived, xgbTree_perdict)

#   xgbTree_perdict

#      0   1

#  0 513  36

#  1  82 260
#Testset prediction

#Convert the categorical variable to Factor before prediction

testset$Pclass <- factor(testset$Pclass)

testset$Sex <- factor(testset$Sex)

testset$Embarked <- factor(testset$Embarked)

testset$Agegroup <- factor(testset$Embarked, exclude=NULL)

testset$company <- factor(testset$company)



#prediction

#rf_perdict_testset <- predict(rf_model, newdata = testset)

#adaboost_perdict_testset <- predict(adaboost_model, newdata = testset)

#xgbTree_perdict_testset <- predict(xgbTree_model, newdata = testset)



#majority voting

#result_combine_test <- data.frame(rf_perdict_testset, adaboost_perdict_testset, xgbTree_perdict_testset)

#majority_test <- apply(result_combine_test, 1, mfv)



#Submission

#submission <- data.frame(PassengerId = testset$PassengerId, Survived = majority_test)

#write.csv(submission, file="submission_20170924.csv", row.names = FALSE)