
# This R script will run on our backend. You can write arbitrary code here!

# Many standard libraries are already installed, such as randomForest
library(randomForest)

# The train and test data is stored in the ../input directory
train <- read.csv("../input/train.csv")
test  <- read.csv("../input/test.csv")

# We can inspect the train data. The results of this are printed in the log tab below
# summary(train)
# str(train)
# table(train$Survived)

# We can inspect the test data. The results of this are printed in the log tab below
# summary(test)
# str(test)
# table(test)

# Impute test$Survived column with 0's
test$Survived <- rep(0,418)

#Assume that all passengers from test data died 
#submit <- data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
#write.csv(submit, file = "Test_all_died.csv", row.names = FALSE)

#Check proportion table between train$Survived & train$Sex attributes
prop.table(table(train$Sex, train$Survived))



# Here we will plot the passenger survival by class
train$Survived <- factor(train$Survived, levels=c(1,0))
levels(train$Survived) <- c("Survived", "Died")
train$Pclass <- as.factor(train$Pclass)
levels(train$Pclass) <- c("1st Class", "2nd Class", "3rd Class")

png("1_survival_by_class.png", width=800, height=600)
mosaicplot(train$Pclass ~ train$Survived, main="Passenger Survival by Class",
           color=c("#8dd3c7", "#fb8072"), shade=FALSE,  xlab="", ylab="",
           off=c(0), cex.axis=1.4)
dev.off()
