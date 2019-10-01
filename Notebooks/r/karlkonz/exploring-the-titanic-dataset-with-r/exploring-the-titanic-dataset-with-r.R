suppressPackageStartupMessages(library(ggplot2))

suppressPackageStartupMessages(library(xgboost))

suppressPackageStartupMessages(library(data.table))

suppressPackageStartupMessages(library(Matrix))

suppressPackageStartupMessages(library(dplyr))

suppressPackageStartupMessages(library(e1071))

suppressPackageStartupMessages(library(party))
set.seed(123456789)

train <- read.csv('../input/train.csv', stringsAsFactors = F)

test  <- read.csv('../input/test.csv', stringsAsFactors = F)



total  <- bind_rows(train, test) 



# Extract title based on proximiy to . in string

total$Title <- gsub('(.*, )|(\\..*)', '', total$Name) 



# Get counts of titles

table(total$Title)



# Titles with very low cell counts to be combined to "rare" level

rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 

                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')



# Also reassign mlle, ms, and mme accordingly

total$Title[total$Title == 'Mlle']        <- 'Miss' 

total$Title[total$Title == 'Ms']          <- 'Miss'

total$Title[total$Title == 'Mme']         <- 'Mrs' 

total$Title[total$Title %in% rare_title]  <- 'Rare'



# Show title counts by sex again

table(total$Title)



total$Embarked <- ifelse(total$Embarked == "", "S", total$Embarked)

#total$Embarked

total$Surname <- sapply(total$Name,  

                      function(x) strsplit(x, split = '[,.]')[[1]][1])

#table(total$Surname)

# Create a family size variable including the passenger themselves

total$Fsize <- total$SibSp + total$Parch + 1



                                                

                          

# Create a family variable 

total$Family <- paste(total$Surname, total$Fsize, sep='_')





# Discretize family size

total$FsizeD[total$Fsize == 1] <- 'NoDep'

total$FsizeD[total$Fsize < 5 & total$Fsize > 1] <- 'mDep'

total$FsizeD[total$Fsize >= 5] <- 'MDep'



                          



total$CabinType <- ifelse(substring(total$Cabin, 1, 1) == 'A', 'A', 

                  ifelse(substring(total$Cabin, 1, 1) == 'B', 'B', 

                  ifelse(substring(total$Cabin, 1, 1) == 'C', 'C', 

                  ifelse(substring(total$Cabin, 1, 1) == 'D', 'D',  

                  ifelse(substring(total$Cabin, 1, 1) == 'E', 'E', 

                  ifelse(substring(total$Cabin, 1, 1) == 'F', 'F',      

                  'zzz'))))))                       

                          

                          
Cabins <- total[which(total$Cabin != '' & total$Cabin != is.na(total$Cabin)), ]

# Find the median of Age for each cabin type

ACabins <- median(Cabins[which(substring(Cabins$Cabin, 1, 1) == 'A'

                                & Cabins$Title != 'Master'), ]$Age, na.rm = TRUE)

BCabins <- median(Cabins[which(substring(Cabins$Cabin, 1, 1) == 'B'

                                & Cabins$Title != 'Master'), ]$Age, na.rm = TRUE)

CCabins <- median(Cabins[which(substring(Cabins$Cabin, 1, 1) == 'C'

                                & Cabins$Title != 'Master'), ]$Age, na.rm = TRUE)

DCabins <- median(Cabins[which(substring(Cabins$Cabin, 1, 1) == 'D'

                                & Cabins$Title != 'Master'), ]$Age, na.rm = TRUE)

ECabins <- median(Cabins[which(substring(Cabins$Cabin, 1, 1) == 'E'

                                & Cabins$Title != 'Master'), ]$Age, na.rm = TRUE)

FCabins <- median(Cabins[which(substring(Cabins$Cabin, 1, 1) == 'F'

                                & Cabins$Title != 'Master'), ]$Age, na.rm = TRUE)

GCabins <- median(Cabins[which(substring(Cabins$Cabin, 1, 1) == 'G'

                                & Cabins$Title != 'Master'), ]$Age, na.rm = TRUE)                         

                    



                       
MasterMedianAge <- median(total[which(total$Title == 'Master'), ]$Age, na.rm = TRUE)
MissingAge <- total[is.na(total$Age), ]                        

# Impute the missing ages for those who have Cabin Fares:



MedianS <- median(subset(total, Embarked == 'S')$Age , na.rm = TRUE)

MedianC <- median(subset(total, Embarked == 'C')$Age , na.rm = TRUE)

MedianQ <- median(subset(total, Embarked == 'Q')$Age , na.rm = TRUE)

                          

MissingAge$Age <- ifelse(substring(MissingAge$Cabin, 1, 1) == 'A'

                                & MissingAge$Title != 'Master', ACabins, 

                  ifelse(substring(MissingAge$Cabin, 1, 1) == 'B'

                                & MissingAge$Title != 'Master', BCabins, 

                  ifelse(substring(MissingAge$Cabin, 1, 1) == 'C'

                                & MissingAge$Title != 'Master', CCabins, 

                  ifelse(substring(MissingAge$Cabin, 1, 1) == 'D'

                                & MissingAge$Title != 'Master', DCabins,  

                  ifelse(substring(MissingAge$Cabin, 1, 1) == 'E'

                                & MissingAge$Title != 'Master', ECabins, 

                  ifelse(substring(MissingAge$Cabin, 1, 1) == 'F'

                                & MissingAge$Title != 'Master', FCabins,  

                  ifelse(substring(MissingAge$Cabin, 1, 1) == 'G'

                                & MissingAge$Title != 'Master', GCabins, 

                  ifelse(is.na(MissingAge$Age) & MissingAge$Embarked == 'S'

                                & MissingAge$Title != 'Master', MedianS,

                  ifelse(is.na(MissingAge$Age) & MissingAge$Embarked == 'C'

                                & MissingAge$Title != 'Master', MedianC,

                  ifelse(is.na(MissingAge$Age) & MissingAge$Embarked == 'Q'

                                & MissingAge$Title != 'Master', MedianQ,

                  MasterMedianAge))))))))))
# Merge the data

CleanTotal <- merge(total[ ,c(1:ncol(total))] , MissingAge[ ,c(1, 6)], by = "PassengerId", all.x = TRUE)

# Merge the 2 Age columns

CleanTotal$Age <- ifelse(is.na(CleanTotal$Age.x), CleanTotal$Age.y, CleanTotal$Age.x)



                          

CleanTotal$Adult <- ifelse(CleanTotal$Age > 15, 1, 0)     

CleanTotal[which(CleanTotal$PassengerId == 1044), ]$Fare <- median(CleanTotal$Fare, na.rm=TRUE)                      

 



CleanTotal$lFare <- log(CleanTotal$Fare + .01)                         

                          #Remove the Age.x and Age.y columns



CleanTotal$lFareRange <- ifelse(CleanTotal$lFare > 3.344, "High",

                         ifelse(CleanTotal$lFare > 2.851, "Med-High",

                         ifelse(CleanTotal$lFare > 2.068, "Med", "Low"

                               )))





names(CleanTotal)



MLTotal <- CleanTotal[ , c("PassengerId", "Survived", "Pclass","Name", 

                             "Sex", "Adult", "SibSp", "Parch", "Ticket",

                             "Embarked", "Surname", "Family",

                             "Fsize", "FsizeD", "CabinType", "Title", "lFareRange", "lFare")]      





LogRegTotal <- CleanTotal[ , c("PassengerId", "Survived", #"Pclass",

                              #"Name", 

                              "Sex", 

                              "Adult", 

                             #"SibSp", "Parch", 

                              "lFare", #"Fsize",  

                              "FsizeD",

                              "CabinType", "Title")]                   



                          

# Read in the training and test data                          

LRtrain <- LogRegTotal[1:891,]

LRtest <- LogRegTotal[892:1309,]

LRtest <- LRtest[, -2]

                          





MLtrain <- MLTotal[1:891,]

MLtest <- MLTotal[892:1309,]

MLtest <- MLtest[, -2]

                     





#names(train)

       

#train_orig <- train

#train <- train[ ,c(1, 2, 3, 4, 5, 8)]

                          

                          

# Check which columns contain NA values

#colnames(train)[colSums(is.na(train)) > 0]



# Examine how many records are missing Age

#nrow(train[which(is.na(train$Age)), ])

                          

                          

                          
str(MLtrain)

str(MLtest)
clnTrn <- suppressWarnings(sapply(data.frame(MLtrain[ ,-c(2, 17)]),as.numeric))

clnTst <- suppressWarnings(sapply(data.frame(MLtest[, -16]),as.numeric))

#clnTrn

clnTrn<-as.matrix(clnTrn, sparse = TRUE)

clnTst<-as.matrix(clnTst, sparse = TRUE)

sparse_matrix_train <- clnTrn

sparse_matrix_test <- clnTst



outputMat = as.data.table(train)



output_vector = outputMat[,Y:=0][Survived == 1,Y:=1][,Y]
bst <- xgboost(data = sparse_matrix_train, label = output_vector, max.depth = 25,

                              eta = .1, nthread = 2, nround = 101,objective = "binary:logistic")



importance <- xgb.importance(feature_names = colnames(sparse_matrix_train), model = bst)



xgb.plot.importance(importance_matrix = importance)
str(MLtrain)

cTrain <- MLtrain[ , -c(1, 4, 9, 11, 12, 13, 15, 18)]



cTrain$Sex <- as.factor(cTrain$Sex)

#cTrain$Name <- as.factor(cTrain$Name)

#cTrain$CabinType <- as.factor(cTrain$CabinType)

#cTrain$Ticket <- as.factor(cTrain$Ticket)

cTrain$Title <- as.factor(cTrain$Title)

cTrain$FsizeD <- as.factor(cTrain$FsizeD)

cTrain$Embarked <- as.factor(cTrain$Embarked)

#cTrain$Surname <- as.factor(cTrain$Surname)

#cTrain$Family <- as.factor(cTrain$Family)

cTrain$lFareRange <- as.factor(cTrain$lFareRange)

str(cTrain)

CondForest <- ctree(Survived ~ ., data = cTrain)#,

plot(CondForest)

#controls = ctree_control(maxsurrogate = 2))
fit <- glm(Survived~ lFare + Adult ,data=LRtrain,family=binomial())

summary(fit)
# Create SVM binary classifier

#svmTrain <- train[ ,-1]

#str(svmTrain)

#svm_model <- svm(as.factor(Survived) ~., data=svmTrain)

#summary(svm_model)
str(MLtest)

cTest <- MLtest[ ,-c(1, 3, 8, 10, 11, 12, 14)]

#str(MLtrain)

cTest$Sex <- as.factor(cTest$Sex)

cTest$Title <- as.factor(cTest$Title)

cTest$FsizeD <- as.factor(cTest$FsizeD)

cTest$Embarked <- as.factor(cTest$Embarked)

#cTrain$Surname <- as.factor(cTrain$Surname)

#cTrain$Family <- as.factor(cTrain$Family)

cTest$lFareRange <- as.factor(cTest$lFareRange)

str(cTest)
# Convert to numeric (by arbitrarily map the characters to numbers.)

#clnTst<-suppressWarnings(sapply(data.frame(test),as.numeric))

#clnTst<-as.matrix(clnTst, spare = TRUE)

#names(test)

#test <- test[ , c("PassengerId", #"Pclass",

#                              #"Name", 

#                              "Sex", 

#                              "Adult", 

#                             #"SibSp", "Parch", 

#                              "lFare", #"Fsize",  

#                              "FsizeD",

#                              "CabinType", "Title")]     



#test$predSVM <- predict(svm_model, test, type = "raw")



#test$boolSVM <- ifelse(test$predSVM >= .5, 1, 0)





test$predLogGLM <- predict(fit, LRtest, type = "response")



test$predCondForest <- predict(CondForest, cTest, type = "response")





#test$predCondForest <- predict(CondForest, cTest)

test$predXGBoost <- predict(bst, clnTst, type = "response")



#confusion.matrix(obs, CleanTest$pred, threshold = 0.5)



#CleanTest$boolLogGLM <- ifelse(test$predLogGLM >= .5, 1, 0)





#head(test$predLogGLM)

#head(test$predXGBoost)

#head(test$predSVM)

#class(test$predSVM)



#test$predSVM <- as.numeric(as.character(test$predSVM))



#class(test$predSVM)

#head(mean(test$predLogGLM, test$predXGBoost, as.numeric(as.character(test$predSVM))))





test$AvgProb <- rowMeans(data.frame(test$predLogGLM, 

                                    #test$predXGBoost, #test$predSVM, 

                                   test$predCondForest))









#test$AvgProb <- sum(test$predLogGLM, test$predXGBoost + test$predSVM)/3

#, (test$predSVM + .01))



#test$boolXGBoost <- ifelse(test$predXGBoost >= .5, 1, 0)



test
class(test$AvgProb)

test$pred <- ifelse(test$AvgProb > .5, 1, 0)

test
# Finish the data.frame() call

my_solution <- data.frame(test$PassengerId, test$pred)



# Next Submit the XGboost model

colnames(my_solution) <- c("PassengerId", "Survived")

# Use nrow() on my_solution

nrow(my_solution)

#nrow(test)



# Finish the write.csv() call

write.csv(my_solution, file = "kaggle.csv", row.names = FALSE)
