#Libraries that are used for model and analysis

library(pROC) # For ROC Curve

library(plyr)

library(dplyr)

library(rpart)  # Decision Tree for Imputing Age

library(randomForest)

library(e1071)

library(ggplot2) # For ploting
train <- read.csv("../input/train.csv", stringsAsFactors=FALSE)

test <- read.csv("../input/test.csv", stringsAsFactors=FALSE)



# lets understand the data

str(train)
table(train$Survived)
prop.table(table(train$Survived))
# Check for missing data

check.missing <- function(x) return(length(which(is.na(x))))

data.frame(sapply(train,check.missing))

data.frame(sapply(test,check.missing))



 # Check blanks

check.blank <- function(x) return(length(which(x == "")))

data.frame(sapply(train, check.blank))

data.frame(sapply(test, check.blank)) 
# Lets add Survived Column to the test data set

test$Survived <- NA



# Combine the test and train

combi <- rbind(train, test)
train$Pclass <- as.factor(train$Pclass)

d <- data.frame(

    class = factor(c('1st class', '2nd class', '3rd class')),

    passengers = c(count(train[train$Pclass == 1,])$n,

          count(train[train$Pclass == 2,])$n,

          count(train[train$Pclass == 3,])$n)

)

d

g <- ggplot(d, aes(x=class, y=passengers))

g <- g + geom_bar(stat='identity', aes(fill=class))

g <- g + labs(title="Passenger class population", x='', y='')

g
train$Survived <- as.factor(train$Survived)

d <- train %>% select(Survived, Pclass)

g <- ggplot(data = d, aes(x=Pclass, fill=Survived))

g <- g + geom_bar() 

g <- g + labs(title="Survivors by passenger class", x="Passenger class", y="Number of passengers") 

g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))

g
train <- tbl_df(train)

train$Name[c(1:5)]
d <- train %>% mutate(AdditionalName = grepl('\\(', Name)) %>% select(Survived, AdditionalName) 

g <- ggplot(data = d, aes(x=AdditionalName, fill=Survived))

g <- g + geom_bar() 

g <- g + labs(title="Survivors with brackets in their names", x="Brackets in name", y="Number of passengers") 

g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))

g
combi <- mutate(combi,AdditionalName = grepl('\\(', Name))

combi$AdditionalName <- as.factor(combi$AdditionalName)

glimpse(combi)
d <- train %>% mutate(QuotedName = grepl('\\"', Name)) %>% select(Survived, QuotedName) 

g <- ggplot(data = d, aes(x=QuotedName, fill=Survived))

g <- g + geom_bar() 

g <- g + labs(title="Survivors with quotes in their names", x="Quotes in name", y="Number of passengers") 

g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))

g
combi <- mutate(combi,QuotedName = grepl('\\(', Name))

combi$QuotedName <- as.factor(combi$QuotedName)

glimpse(combi)
#Convert names column to characters

train$Name <- as.character(train$Name)



train$Title <- sapply(train$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})



# Title has space so lets remove it

train$Title <- sub(' ', '', train$Title)



#Lets understand the title

table(train$Title)
train$Title[train$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'

train$Title[train$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'

train$Title[train$Title %in% c('Mlle', 'Ms')] = 'Miss'

train$Title[train$Title == 'Mme'] = 'Mrs'

train$Title <- as.factor(train$Title)
d <- train %>%

    select(Survived, Title) %>%

    group_by(Title, Survived) %>% 

    summarise(n = n()) %>% 

    mutate(rate = n/sum(n) * 100) %>% 

    filter(Survived == 1)

g <- ggplot(data = d, aes(x=Title, y=rate))

g <- g + geom_bar(stat='identity') 

g <- g + labs(title="Survival rate by title", x="Title", y="Survival rate in %") 

g
combi$Name <- as.character(combi$Name)

combi$Title <- sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})

# Title has space so lets remove it

combi$Title <- sub(' ', '', combi$Title)

combi$Title[combi$Title %in% c('Capt', 'Don', 'Major', 'Sir','Col','Dr')] <- 'Sir'

combi$Title[combi$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'

combi$Title[combi$Title %in% c('Mlle', 'Ms')] = 'Miss'

combi$Title[combi$Title == 'Mme'] = 'Mrs'

combi$Title <- as.factor(combi$Title)



table(combi$Title)
(train %>% filter(Cabin == "") %>% count)$n
d <- train %>%

    filter(Cabin != "") %>%

    mutate(Cabin = gsub("^(\\w).*$", "\\1", Cabin)) %>%

    select(Cabin, Survived)



g <- ggplot(data = d, aes(x=Cabin, fill=Survived))

g <- g + geom_bar(width=0.5) 

g <- g + labs(title="Survivors by Cabin", x="Cabin floor", y="Number of passengers") 

g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))

g
library(tidyr)

dd <- train %>% 

    group_by(Cabin, Survived) %>% 

    summarise(n = n()) %>%

    mutate(survivalrate = n / sum(n)) %>%

    select(Survived, Cabin, survivalrate) %>%

    spread(Survived, survivalrate)

colnames(dd) <-  c('floor', 'died', 'survived')

dd
d <- train %>% select(Survived, SibSp)

g <- ggplot(data = d, aes(x=SibSp, fill=Survived))

g <- g + geom_bar() 

g <- g + labs(title="Survivors with siblings", x="Number of siblings", y="Number of passengers") 

g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))

g
d <- train %>% select(Survived, Parch)

g <- ggplot(data = d, aes(x=Parch, fill=Survived))

g <- g + geom_bar() 

g <- g + labs(title="Survivors with Parents/Children", x="Number of relation", y="Number of passengers") 

g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))

g
d <- train %>% 

    select(Sex, Survived, Parch) %>% 

    filter(Sex == 'male') %>% 

    select(Survived, Parch) %>% 

    group_by(Parch, Survived) %>% 

    summarise(n=n()) %>% 

    mutate(rate=n/sum(n) * 100)

d
d <- combi %>% 

    select(Sex, Survived, Parch) %>% 

    filter(Sex == 'male') %>% 

    select(Survived, Parch) %>% 

    group_by(Parch, Survived) %>% 

    summarise(n=n()) %>% 

    mutate(rate=n/sum(n) * 100)



combi$MenWithMoreRel <- 0

combi$MenWithMoreRel[combi$Parch>2 & combi$Sex=='Male'] <- 1

glimpse(combi)
# Family size column

combi$FamilySize <- 0 

combi$FamilySize <- combi$SibSp + combi$Parch + 1
d <- train %>% select(Age, Survived) %>% filter(!is.na(Age))

g <- ggplot(data = d, aes(x=Age, fill=Survived))

g <- g + geom_bar() 

g <- g + labs(title="Age of Survivors", x="Age", y="Number of passengers") 

g <- g + scale_fill_discrete(name="Survived", labels=c("no", "yes"))

g
(train %>% filter(is.na(Age)) %>% count)$n
glimpse(combi)

check.missing <- function(x) return(length(which(is.na(x))))

data.frame(sapply(combi,check.missing))
combi$Fare[1044] <- median(combi[combi$Pclass == '3' & combi$Embarked == 'S', ]$Fare, na.rm = TRUE)

combi$Embarked <- as.factor(combi$Embarked)

combi$FamilySize <- as.factor(combi$FamilySize)



table(combi[is.na(combi$Age) | combi$Age == 0,"Pclass"]) # Lots of class 3 missing compared to the rest

boxplot(Age ~ Pclass, data=combi, main="Age vs Class", xlab="Class", ylab="Fare") # Looks like ages are are somewhat indicative of class so use this to impute

library(mice)

glimpse(train)

mice_mod <- mice(train[, !names(train) %in% c('PassengerId','Name',

                                              'AdditionalName','QuotedName','MenWithMoreRel',

                                              'Ticket','Cabin','Survived')], method='rf') 



# Save the complete output 

mice_output <- complete(mice_mod)

train$Age2 <- mice_output$Age

train$Age2 <- as.integer(train$Age2)
d <- train %>% select (PassengerId,Parch,Name) %>% filter(Parch>2) %>% 

mutate('familyname' = gsub("^([^,]*),.*", "\\1", Name, perl=TRUE))

train <- left_join(train,d)



train$Mother <- 'Not Mother'

train$Mother[train$Sex == 'female' & train$Parch > 0 & train$Age > 18 & train$Title != 'Miss'] <- 'Mother'

train$Mother <- as.factor(train$Mother)



 train %>% 

 select(Fare, FamilySize) %>% 

 group_by(FamilySize) %>%

 summarise(n=sum(Fare)) %>% 

 mutate(rate=n/FamilySize)

combi$Sex <- as.factor(combi$Sex)



#train <- combi[1:891,]

k = 5 #Folds

train$Title <- as.factor(train$Title)



train$Cabin2 <-substr(train$Cabin, 0, 1)

train$Cabin2 <- as.factor(train$Cabin2)

train$id <- sample(1:k, nrow(train), replace = TRUE)

list <- 1:k







# prediction and testset data frames that we add to with each iteration over

# the folds



prediction <- data.frame()

testsetCopy <- data.frame()



#Creating a progress bar to know the status of CV

progress.bar <- create_progress_bar("text")

progress.bar$init(k)



for (i in 1:k){

  # remove rows with id i from dataframe to create training set

  # select rows with id i to create test set

  trainingset <- subset(train, id %in% list[-i])

  testset <- subset(train, id %in% c(i))

  # run a random forest model

  mymodel <- randomForest(

      as.factor(Survived) ~ Pclass  + Age2 + Fare + 

                                   Embarked + Title + FamilySize + Sex + Cabin2 + Mother ,

                                   data=trainingset,

                                   importance=T,

                                   ntrees=100)

                                                     

  # remove response column 1, Sepal.Length

  temp <- as.data.frame(predict(mymodel, testset[,-1]))

  # append this iteration's predictions to the end of the prediction data frame

  prediction <- rbind(prediction, temp)

  

  # append this iteration's test set to the test set copy data frame

  # keep only the Sepal Length Column

  testsetCopy <- rbind(testsetCopy, as.data.frame(testset[c('Survived')]))

  

  progress.bar$step()

}



# add predictions and actual Sepal Length values

result <- cbind(prediction, testsetCopy)

names(result) <- c("Predicted", "Actual")

table(result)



#confusionMatrix(result$Survived, result$Predicted)

# As an example use Mean Absolute Error as Evalution 

#summary(result$Difference)


combi$Sex <- as.factor(combi$Sex)

train <- combi[1:891,]

test <- combi[892:1309,]


glimpse(train)

set.seed(415)

randomForest_model <- randomForest(as.factor(Survived) ~ Pclass + AdditionalName + 

                                   MenWithMoreRel + QuotedName + Age + SibSp + Parch + Fare + 

                                   Embarked + Title + FamilySize + Sex,

                                   data=train,

                                   importance=T,

                                   ntrees=250)

varImpPlot(randomForest_model)

getTree(randomForest_model, 1)
predictedY <- predict(randomForest_model, test,type="class")

submit <- data.frame(PassengerId = test$PassengerId, Survived = predictedY)

submit

write.csv(submit, file = "svmTree.csv", row.names = FALSE)