# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



# Any results you write to the current directory are saved as output.
library(dplyr) #Data Manipulation

library(rpart) 

library(rpart.plot)

library(randomForest)

library(ggplot2)

library(ggthemes)

train <- read.csv("../input/train.csv", stringsAsFactors= F)

test <- read.csv("../input/test.csv", stringsAsFactors= F)



full_data <-bind_rows(train,test)

str(full_data)
full_data$AgeRange[full_data$Age <= 9] <- 'lessthan9'

full_data$AgeRange[full_data$Age <= 25 & full_data$Age> 9] <- 'bw9and25' 

full_data$AgeRange[full_data$Age <=50 & full_data$Age > 25] <- 'bw25and50' 

full_data$AgeRange[full_data$Age > 50] <- 'greaterthan50'

full_data$Title <- sapply(full_data$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})

full_data$Title <- sub(' ', '', full_data$Title)

#table(full_data$Title)

#table(full_data$Title, full_data$Sex)



full_data$Title[full_data$Title == 'Mlle']        <- 'Miss' 

full_data$Title[full_data$Title == 'Ms']          <- 'Miss'

full_data$Title[full_data$Title == 'Mme']         <- 'Mrs'



high_rank_male <- c('Capt','Col','Don','Jonkheer','Rev','Sir')

high_rank_female <- c('Dona','Lady','the Countess')





full_data$Title[full_data$Title %in% high_rank_male] <- 'High_rank_male'

full_data$Title[full_data$Title %in% high_rank_female] <- 'High_rank_female'
table(full_data$Title)
#table(full_data$Fare)

full_data$FareRange[full_data$Fare < 10] <- 'lessthan10'

full_data$FareRange[full_data$Fare <20 & full_data$Fare >= 10] <- 'bw10and20'

full_data$FareRange[full_data$Fare <30 & full_data$Fare >= 20] <- 'bw20and30'

full_data$FareRange[full_data$Fare >= 30] <- 'morethan30'

table(full_data$FareRange)

full_data$familysize = full_data$SibSp + full_data$Parch + 1

table(full_data$familysize)
full_data$Sex = as.factor(full_data$Sex)

full_data$FareRange = as.factor(full_data$FareRange)

full_data$AgeRange = as.factor(full_data$AgeRange)

mod_train <- full_data[1:891,]

mod_test <- full_data[892:1309,]
str(mod_train)
Tree1 <- rpart(Survived ~ Pclass + Sex + AgeRange + FareRange + Embarked + Title + familysize + SibSp + Parch,

               data=mod_train, 

               method="class",

              control=rpart.control(minsplit=2, cp=0))
prp(Tree1)
Tree1Prediction <- predict(Tree1, mod_test, type = "class")

submit <- data.frame(PassengerId = mod_test$PassengerId, Survived = Tree1Prediction)

write.csv(submit, file = "Tree1Prediction.csv", row.names = FALSE)
table(mod_test$AgeRange, Tree1Prediction)
aggregate(Survived ~ FareRange + Pclass + Sex, data=mod_train, FUN=function(x) {sum(x)/length(x)})

mod_test2 <-mod_test

table(Tree1Prediction)

mod_test2$Survived <- Tree1Prediction

str(mod_test2)

#aggregate(Survived ~ FareRange + Pclass + Sex, data=mod_test2, FUN=function(x) {sum(x)/length(x)})



