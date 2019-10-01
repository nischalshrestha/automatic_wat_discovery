library(caret)

library(randomForest)

train<- read.csv(file="../input/train.csv",stringsAsFactors = T)

test <- read.csv(file="../input/test.csv",stringsAsFactors = T)

#Creating new features

train$estAge <- grepl("\\.5",train$Age)

test$estAge <- grepl("\\.5",test$Age)



#Extracting Title from the Name

train$Title <- gsub('(.*, )|(\\..*)', '', train$Name)

test$Title <- gsub('(.*, )|(\\..*)', '', test$Name)



train$Title[train$Title == 'Mlle']        <- 'Miss' 

train$Title[train$Title == 'Ms']          <- 'Miss'

train$Title[train$Title == 'Mme']         <- 'Mrs' 



test$Title[test$Title == 'Mlle']        <- 'Miss' 

test$Title[test$Title == 'Ms']          <- 'Miss'

test$Title[test$Title == 'Mme']         <- 'Mrs' 



MainTitles <- c("Master","Miss","Mr","Mrs","Rev")



train[!train$Title %in% MainTitles,c("Title")] <- "Others"

test[!test$Title %in% MainTitles,c("Title")] <- "Others"



train$Title <- as.factor(train$Title)

train$Embarked <- as.factor(train$Embarked)



train$Cabinsize<- nchar(as.character(train$Cabin))==0



test$Title <- as.factor(test$Title)

test$Embarked <- as.factor(test$Embarked)



test$Cabinsize<- nchar(as.character(test$Cabin))==0



colnames(train)

var <- c("Pclass","Sex","Age","SibSp",'Parch','Fare','Embarked','estAge','Title','Cabinsize')

ImputeTrain <- train[,var]

colnames(ImputeTrain)



preProcValues <- preProcess(ImputeTrain, method = c("knnImpute"))

train <- predict(preProcValues, train)

test <- predict(preProcValues,test)



summary(train)



# Set a random seed

set.seed(754)



# Build the model (note: not all possible variables are used)

rf_model <- randomForest(factor(Survived) ~ Pclass + Title + estAge + (SibSp + Parch) + Sex + Age + Cabinsize + Fare ,

                                           data = train)



# Show model error

plot(rf_model, ylim=c(0,0.36))

legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)



prediction <- predict(rf_model,test)

# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)

solution <- data.frame(PassengerID = test$PassengerId, Survived = prediction)



# Write the solution to file

write.csv(solution, file = 'rf_mod_Solution.csv', row.names = F)
















