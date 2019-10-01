library(dplyr)



# Read the raw data

train <- read.csv("../input/train.csv")

test <- read.csv("../input/test.csv")



# Look at the dimensions of the datasets

dim(train)

dim(test)



# Generate a combined dataset for cross validation

# Note initial errors about coercing factors to characters

full <- bind_rows(train, test)

dim(full)



head(full)
# These don't seem to work, as the Kaggle kernel doesn't allow insertion of arbirary HTML from

# packages like googleVis.



# suppressPackageStartupMessages(library(googleVis))

# gvisTable(train)
str(train)

head(train)
str(test)

head(test)
library(rpart)
decision_tree <- rpart(

    Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, 

    data = train, 

    method = "class")



# Load the packages to build the fancy plot

library(rattle)

library(rpart.plot)

library(RColorBrewer)



# Plot the tree

fancyRpartPlot(decision_tree)
prediction_1 <- predict(decision_tree, newdata = test, type = "class")

solution_1 <- data.frame(PassengerId = test$PassengerId, Survived = prediction_1)

write.csv(solution_1, file = "my_solution.csv", row.names = FALSE)
# summary() will give me NAs for numerical values, but we need to treat NA for strings as empty strings

summary(full)



# identify passenger without fare

id <- full[which(is.na(full$Fare)), 1]

full[id,]



# compute what the fare should be by computing the median fare of 3rd class passengers who left from

# Southhampton



median_fare <- full %>%

    filter(Pclass == '3' & Embarked == 'S') %>%

    summarise(missing_fare = median(Fare, na.rm = TRUE))



median_fare
# characters need special handling as well because it is not obvious

full$Embarked[full$Embarked == ""] <- NA

full[which(is.na(full$Embarked)), 1]



full$Cabin[full$Cabin == ""] <- NA

full[which(is.na(full$Cabin)), 1]
library(caret)



controlParameters <- trainControl(

    method = "cv",

    number = 10,

    repeats = 10,

    verboseIter = TRUE

)



decision_tree_model <- train(

    Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, 

    data = full,

    trControl = controlParameters,

    method = "rpart",

    na.action = na.omit

)

train_1 <- data.frame(train$Pclass, train$Sex, train$Age, train$SibSp, train$Parch, train$Fare, train$Embarked)

head(train_1)
plot(train)
summary(solution_1)