library(randomForest)

library(ggplot2)

library(readr) # CSV file I/O, e.g. the read_csv function

library(dplyr)

library(mice)



# Input data files are available in the "../input/" directory.
train <- read.csv( "../input/train.csv" )

test <- read.csv( "../input/test.csv" )



full <- bind_rows( train, test )



full$Survived <- factor( full$Survived )
prop.table( table( full$Sex, full$Survived ) )
full$IsChild <- "Adult"

full$IsChild[train$Age < 18] <- "Child"

full$IsChild <- factor(full$IsChild)



prop.table( table( full$IsChild, full$Survived ) )
ggplot( full, aes(x = Sex, fill = factor(Survived))) +

    geom_bar(stat='count', position='dodge') +

    labs(x = 'Sex')
md.pattern( full )
train <- full[1:891,]

test <- full[892:1309,]



set.seed(46)



rf_model <- randomForest( Survived ~ Pclass + Sex +

                                     SibSp + Parch + 

                                     IsChild,

                                     data = train )



plot(rf_model)

legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)
prediction <- predict( rf_model, test )



solution <- data.frame( PassengerId = test$PassengerId, Survived = prediction )



write.csv( solution, file = 'rf_solution.csv', row.names = F )