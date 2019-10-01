# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library('ggplot2') # Data visualization

library('readr') # CSV file I/O, e.g. the read_csv function

library('scales') # visualization

library('dplyr') # data manipulation

library('randomForest') # classification algorithm

library('rattle') # visualization

library('rpart.plot') # visualization

library('RColorBrewer') # visualization

library('party') # classification algorithm
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")

train = read_csv('../input/train.csv')

test = read_csv('../input/test.csv')

missing.types <- c("NA", "")

train.column.types <- c('integer',   # PassengerId

                        'factor',    # Survived 

                        'factor',    # Pclass

                        'character', # Name

                        'factor',    # Sex

                        'numeric',   # Age

                        'integer',   # SibSp

                        'integer',   # Parch

                        'character', # Ticket

                        'numeric',   # Fare

                        'character', # Cabin

                        'factor'     # Embarked

)

test.column.types <- train.column.types[-2]

# Any results you write to the current directory are saved as output.
#CLEANING THE DATASET



#Combining both datasets for ease of cleaning the data

test$Survived<- NA 

combi<-rbind(train,test)



#Lastnames : I found quit a few differnt last name which would have made our decision trees much

#more complex. One important thing to consider for name that certain titles can be associated to

#higher class passengers



#converting name to character

combi$Name <- as.character(combi$Name)



#seprating the title from rest of the name

combi$Title <- sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})

combi$Title <- sub(' ', '', combi$Title)

#table(combi$Title)



#combining similar titles

combi$Title[combi$Title %in% c('Mlle','Mme')]<-'Mlle'

combi$Title[combi$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'

combi$Title[combi$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'



#converting title back to factor

combi$Title <- factor(combi$Title)



#Familysize : Family size could be one of the factor which affect the survival rate. If the family

#size is considerably large the chances of leaving some of the family member behind increased.



#calculating family size by adding number of sibling/spouse and number of parent/child with 1 for self

combi$FamilySize <- combi$SibSp+combi$Parch+1



#a better picture can be drwan if we have the family surname with the family size

combi$Surname <- sapply(combi$Name, FUN=function(x){strsplit(x, split='[,.]')[[1]][1]})



#creating a new column FamilyId to associate family surname with number of members

combi$FamilyID <- paste(as.character(combi$FamilySize), combi$Surname, sep="")



#Family ID have lot of different entries. So to bring down the variable we converted all the value

#with family size less then 2 to small

combi$FamilyID[combi$FamilySize <= 2] <- 'Small'



#but we still notice some of the FamilyID escape our branding

#table(combi$FamilyID)

famIDS <- data.frame(table(combi$FamilyID))

famIDS <- famIDS[famIDS$Freq <= 2,]

combi$FamilyID[combi$FamilyID %in% famIDS$Var1] <- 'Small'

combi$FamilyID <- factor(combi$FamilyID)



#for applying random forest there are few constraints. one of the first thing is random forest do not

#process n/a values so we have to convert all n/a values



#the age variable has close to 20% of the entries as n/a so we predict age based on other variables

#we use the "anova" method because the value is continues

Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize,

                data=combi[!is.na(combi$Age),], 

                method="anova")

combi$Age[is.na(combi$Age)] <- predict(Agefit, combi[is.na(combi$Age),])



#similarly we find missing values in embark and replace it 

#summary(combi)

#summary(combi$Embarked)

which(combi$Embarked == '')

combi$Embarked[c(62,830)] = "S"

combi$Embarked <- factor(combi$Embarked)



#replace the missing value in Fare

#summary(combi$Fare)

which(is.na(combi$Fare))

combi$Fare[1044] <- median(combi$Fare, na.rm=TRUE)



#after going through the dataset I still found quite many different familyID so to reduce the number

#all family with less than or equal to 3 member would be categorized as "Small" family

combi$FamilyID2 <- combi$FamilyID

#summary(combi$FamilyID2)

combi$FamilyID2 <- as.character(combi$FamilyID2)

combi$FamilyID2[combi$FamilySize <= 3] <- 'Small'

combi$FamilyID2 <- factor(combi$FamilyID2)



#seprating train and test data to run random forest

train <- combi[1:891,]

test <- combi[892:1309,]
#applying random forest technique



set.seed(415)#set the seed to avoid future confusion



fit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare +

                    Embarked + Title + FamilySize + FamilyID2,

                    data=train, 

                    importance=TRUE, 

                    ntree=2000)

varImpPlot(fit)



Prediction <- predict(fit, test)

random <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)

random
set.seed(415)

fit <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare +

                Embarked + Title + FamilySize + FamilyID,

                data = train, 

                controls=cforest_unbiased(ntree=2000, mtry=3))

Prediction <- predict(fit, test, OOB=TRUE, type = "response")

submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)

submit