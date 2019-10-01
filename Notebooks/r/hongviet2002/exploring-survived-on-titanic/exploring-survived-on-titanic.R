# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function

library(rpart)

library(rpart.plot)	



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#setwd("..\\DataMining\\LeViet_Work")

train <- read.csv("../input/train.csv")

test<-read.csv("../input/test.csv")

test$Survived <- 0



New_Data <- rbind(train, test)

#View(New_Data)

write.csv(New_Data,"New_Data_LeViet.csv")

New_Data$Name <- as.character(New_Data$Name)

New_Data$Title <- sapply(New_Data$Name, FUN=function(x) {strsplit(x, split="[,.]")[[1]][2]})

New_Data$Title <- sub(' ', '', New_Data$Title)

New_Data$Title[New_Data$PassengerId == 797] <- 'Mrs' # female doctor

New_Data$Title[New_Data$Title %in% c('Lady', 'the Countess', 'Mlle', 'Mee', 'Ms')] <- 'Miss'

New_Data$Title[New_Data$Title %in% c('Capt', 'Don', 'Major', 'Sir', 'Col', 'Jonkheer', 'Rev', 'Dr', 'Master')] <- 'Mr'

New_Data$Title[New_Data$Title %in% c('Dona')] <- 'Mrs'

New_Data$Title <- factor(New_Data$Title)

New_Data$Embarked[c(62,830)] = "S"

New_Data$Embarked <- factor(New_Data$Embarked)

New_Data$Fare[1044] <-median(New_Data$Fare, na.rm = TRUE)

New_Data$family_size <- New_Data$SibSp + New_Data$Parch + 1

write.csv(New_Data,"New_Data_LeViet_02.csv")

#View(New_Data)

predicted_age <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + family_size,

                       data=New_Data[!is.na(New_Data$Age),], method="anova")



predicted_age <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + family_size,

                       data=New_Data[!is.na(New_Data$Age),], method="anova")

New_Data$Age[is.na(New_Data$Age)] <- predict(predicted_age, New_Data[is.na(New_Data$Age),])

write.csv(New_Data,"New_Data_LeViet_02.csv")

train_new_LeViet <- New_Data[1:891,]

test_new_Leviet <- New_Data[892:1309,]

#View(train_new_LeViet)

#View(test_new_Leviet)



test_new_Leviet$Survived <- NULL

#View(test_new_Leviet)

train_new_LeViet$Cabin <- substr(train_new_LeViet$Cabin,1,1)

test_new_Leviet$Cabin <- substr(test_new_Leviet$Cabin,1,1)



train_new_LeViet$Cabin[train_new_LeViet$Cabin == ""] <- "H"

test_new_Leviet$Cabin[test_new_Leviet$Cabin == ""] <- "H"



train_new_LeViet$Cabin[train_new_LeViet$Cabin == "T"] <- "H"



train_new_LeViet$Cabin <- factor(train_new_LeViet$Cabin)

test_new_Leviet$Cabin <- factor(test_new_Leviet$Cabin)



str(train_new_LeViet)

str(test_new_Leviet)



LeViet_Tree <- rpart(Survived ~ Age + Sex + Pclass  + family_size, data = train_new_LeViet, method = "class", control=rpart.control(cp=0.0001))



summary(LeViet_Tree)





prp(LeViet_Tree, type = 4, extra = 100)



LeViet_prediction <- predict(LeViet_Tree, test_new_Leviet, type = "class")

head(LeViet_prediction)



vector_passengerid <- test_new_Leviet$PassengerId



LeViet_Solution <- data.frame(PassengerId = vector_passengerid, Survived = LeViet_prediction)



head(LeViet_Solution)



write.csv(LeViet_Solution, file = "LeViet_Solution.csv",row.names=FALSE)
