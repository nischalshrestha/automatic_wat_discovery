##Loading the required libraries and data sets. 

library(tidyverse) #visualization and data wrangling

library(Amelia) #Visualize N/As

library(randomForest) #random forest models

library(caTools) #train/test spilt



train <- read.csv("../input/train.csv")

test <- read.csv("../input/test.csv")
str(train)

summary(train)
any(is.na(train)) #Are there NA values?

missmap(train, legend = FALSE)
ggplot(train, aes(x = as.factor(Pclass), y = Age)) + geom_boxplot(aes(fill = as.factor(Pclass))) + 

                                                    labs(title = "Age According to Ticket Classes",

                                                           x = "Ticket Classes", y = "Age")
anovamod <- lm(Age ~ as.factor(Pclass), data = train) #model for ANOVA

anova(anovamod) #presenting the ANOVA table

plot(anovamod) #plotting required graphs such as residuals vs fitted and QQ-plots
first_class <- filter(train, Pclass == 1)

floor(mean(first_class[,6], na.rm = T))

second_class <- filter(train, Pclass == 2)

floor(mean(second_class[,6], na.rm = T))

third_class <- filter(train, Pclass == 3)

floor(mean(third_class[,6], na.rm = T))

#results came out to be 38, 29 and 25



for (i in (1:nrow(train))) {

    if (is.na(train[i,6]) == TRUE){

        if (train[i,3] == 1){

            train[i,6] <- 38

        } else if (train[i,3] == 2){

            train[i,6] <- 29

        } else if (train[i,3] == 3){

            train[i,6] <- 25

        } 

    } else {

        train[i,6] <- train[i,6]

    }

}
missmap(train, legend = FALSE)
train <- select(train, -PassengerId, -Name, -Ticket, -Cabin) 

set.seed(101) 

split <- sample.split(train$Survived, SplitRatio = 0.7) #this is from the caTools library 

train_training <- subset(train, split == TRUE)

train_test <- subset(train, split == FALSE)
logistic_model <- glm(Survived ~.,family = binomial(link = "logit"), data = train_training)

summary(logistic_model)

step(logistic_model)
logistic_model_filtered <- glm(Survived~ Pclass + Sex + Age + SibSp + Fare, family = binomial(link = "logit"), data = train_training)

summary(logistic_model_filtered)
test_predictions <- predict(logistic_model_filtered, newdata = train_test, type = 'response')

table(train_test$Survived, test_predictions > 0.5)
test_predictions_results <- ifelse(test_predictions > 0.5,1,0) #converting true-false values into 1s and 0s

misClasificError_logit <- mean(test_predictions_results != train_test$Survived)

print(paste('Accuracy',1-misClasificError_logit)) #0.774
rf_model <- randomForest(as.factor(Survived)~.,data = train_training, importance = TRUE)

print(rf_model)

rf_model$importance

rf_model$confusion
rf_test_prediction <- predict(rf_model, train_test)

table(rf_test_prediction,train_test$Survived)

misClasificError_rf <- mean(rf_test_prediction != train_test$Survived)

print(paste('Accuracy',1-misClasificError_rf)) #~0.828
missmap(test, legend = FALSE)
first_class_test <- filter(test, Pclass == 1)

floor(mean(first_class_test[,5], na.rm = T))

second_class_test <- filter(test, Pclass == 2)

floor(mean(second_class_test[,5], na.rm = T))

third_class_test <- filter(test, Pclass == 3)

floor(mean(third_class_test[,5], na.rm = T))

#results came out to be 40, 28 and 24



for (i in (1:nrow(test))) {

  if (is.na(test[i,5]) == TRUE){

    if (test[i,2] == 1){

      test[i,5] <- 40

    } else if (test[i,2] == 2){

      test[i,5] <- 28

    } else if (test[i,2] == 3){

      test[i,5] <- 24

    } 

  } else {

    test[i,5] <- test[i,5]

  }

}
missmap(test, legend = FALSE)
PassengerId <- test[,1] #saving the passengerid column for later

test <- test[,-c(1,10,3,8)] #removing columns

str(test) #check str of test

str(train) #check str of train
(which(is.na(train$Embarked == ""))) 
train <- train %>%

            filter(Embarked != "") %>%

            droplevels()

#we check the str of train again

str(train)
## Now we run the entire model on the train data set

test_model_final <- randomForest(as.factor(Survived)~.,data = train, importance = TRUE)



## Run predictions

test_model_final_predictions <- predict(test_model_final, test, predict.all = TRUE)



#Exporting predictions into a file 

results_final <- as.vector(test_model_final_predictions$aggregate)

results_frame <- data.frame(PassengerId, results_final)

colnames(results_frame) <- c("PassengerId", "Survived")



which(is.na(results_frame$Survived == T)) #check for N/A values
results_frame[153,2] <- 0

# Exporting to file.

#write.table(results_frame, file = "Submission.csv", sep = ",", row.names = FALSE)

#View(results_frame)