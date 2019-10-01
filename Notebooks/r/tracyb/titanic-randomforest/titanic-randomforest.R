# Load packages and read in data

library(ggplot2)

library(rpart)

library(randomForest)

train <- read.csv("train.csv")

test <- read.csv("test.csv")



# Combine train and test datasets

test$Survived <- NA

full_data <- rbind(train, test)



# Feature engineering

# Create a family size variable

full_data$familysize <- NA

full_data$familysize <- full_data$SibSp + full_data$Parch + 1

# Create a child variable

full_data$Child <- NA

full_data$Child[full_data$Age >= 18] <- 0

full_data$Child[full_data$Age < 18] <- 1

#Create a title variable

full_data$Title <- gsub('(.*, )|(\\..*)', '', full_data$Name)

table(full_data$Sex, full_data$Title)

rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 

                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')

full_data$Title[full_data$Title == 'Mlle']        <- 'Miss' 

full_data$Title[full_data$Title == 'Ms']          <- 'Miss'

full_data$Title[full_data$Title == 'Mme']         <- 'Mrs' 

full_data$Title[full_data$Title %in% rare_title]  <- 'Rare Title'

full_data$Title <- factor(full_data$Title)



# Find missing values

summary(full_data)

summary(full_data$Embarked)

full_data[c(1:1309), 'Embarked']

full_data[c(62, 830),]

# Impute missing values

full_data[c(62, 830), 'Fare']

ggplot(full_data, aes(x = Embarked, y = Fare, Fill = factor(Pclass))) + geom_boxplot()

full_data$Embarked[c(62, 830)] <- 'C'

full_data[c(1044), 'Fare']

full_data[1044, ]

ggplot(full_data[full_data$Pclass == '3' & full_data$Embarked == 'S', ], aes(x=Fare)) + geom_density()

median(full_data[full_data$Pclass == '3' & full_data$Embarked == 'S', ]$Fare, na.rm = TRUE)

full_data$Fare[1044] <- 8.05

# Impute missing ages using rpart

age_tree <- rpart(Age ~ Title + Pclass + Sex + Fare + Embarked, data = full_data[!is.na(full_data$Age),], method = "anova")

full_data$Age[is.na(full_data$Age)] <- predict(age_tree, full_data[is.na(full_data$Age),])

full_data$Child[full_data$Age >= 18] <- 0

full_data$Child[full_data$Age < 18] <- 1



# Break up train and test datasets

train <- full_data[1:891,]

test <- full_data[892:1309,]



# Make predictions with randomforest

set.seed(800)

rf_model <- randomForest(factor(Survived) ~ Pclass + Title + Sex + SibSp + Age + Parch + Fare + Embarked + Child + familysize, data=train)

plot(rf_model)

prediction <- predict(rf_model, test)

# Export solution as csv

solution <- data.frame(PassengerId = test$PassengerId, Survived = prediction)

write.csv(solution, file = 'rf_mod_solution.csv', row.names = FALSE)