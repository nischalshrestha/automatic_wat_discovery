## Input and prepare the data for modelling



# Input data files and combine into one data frame

train <- read.csv("../input/train.csv")

test <- read.csv("../input/test.csv")



# Combine the data files because we want to use any extra information in the training data that 

# we can glean from the test data set

train$Set <- rep("Train", nrow(train))

test$Set <- rep("Test", nrow(test))

test$Survived <- NA

titanic <- rbind(train, test)



# After some exploratory analysis it appears that families and perhaps other groups travelling

# together did so on the same ticket. Using Ticket as a group identifier therefore is useful

# given that people may have looked for others in their group before hitting the life boats

# (and therefore their survival would be correlated) and also some missing data may be estimated

# from others travelling in the same group.



# Firstly let's add the size of the group to the data set

titanic$NGrp <- table(titanic$Ticket)[titanic$Ticket]



# From the non-missing Cabin information we can see that people travelling in groups were almost

# always located on the same deck and so let's use the cabin information from everyone in the

# group to assign a deck to individuals even when this information is otherwise missing.

titanic$Deck <- tapply(substr(as.character(titanic$Cabin),1,1),titanic$Ticket,max)[titanic$Ticket]



# We can add variables for the known outcomes for others in a passenger's group (taking care not

# to add the outcome of the passenger!) as this may be a useful predictor under the hypethesis

# that survival is correlated within groups.

# Below K=Known, S=Survived, D=Died, IG=In Group, UO=Unknown Outcome, ESR=Estimated Survival Rate

attach(titanic)

sumNotNA <- function(x) sum(x[!is.na(x)])

titanic$KSIG <- tapply(Survived, Ticket, sumNotNA)[Ticket] - ifelse(is.na(Survived),0,Survived)

titanic$KDIG <- tapply(1-Survived, Ticket, sumNotNA)[Ticket] - ifelse(is.na(Survived),0,1-Survived)

titanic$UOIG <- tapply(Survived, Ticket, function(x) sum(is.na(x)))[Ticket] - is.na(Survived)

titanic$ESRIG <- titanic$KSIG/(titanic$KSIG+titanic$KDIG)

detach()



# Let's add the passenger's title to the data set, recoding French, Italian and Dutch titles

# to English and combining army officers together, and nobility together.

Title <- substring(lapply(strsplit(as.character(titanic$Name),", "),"[",2),1,4)

Title[Title=="Capt"|Title=="Majo"|Title=="Col."] <- "Army"

Title[Title=="Don."] <- "Mr. "

Title[Title=="Dona"] <- "Mrs."

Title[Title=="Mlle"] <- "Miss"

Title[Title=="Mme."] <- "Mrs."

Title[Title=="Ms. "] <- "Miss"

Title[Title=="the "] <- "Lady"

Title[Title=="Jonk"] <- "Sir."

titanic$Title <- Title

rm(Title)
## Modelling



# Here we will use CART (implemented in rPart) to model survival and predict the outcomes in the

# test data. CART is good in that: it has good default stopping rules to not overfit the data;

# it handles missing data very well; and it can handle many predictors and highly-correlated

# predictors. The resulting model of CART is also often easy to understand and convey.



library(rpart)

fm <- rpart(Survived ~ Sex + NGrp + Pclass + Age + SibSp + Parch + Fare + Deck + Title + Embarked

            + KSIG + KDIG + UOIG + ESRIG, data=titanic, subset=titanic$Set=="Train", method="class")

pred <- round(predict(fm, newdata=titanic)[,2])



# And write out the results for subission

submission <- data.frame(PassengerId=titanic$PassengerId, Survived=pred)[titanic$Set=="Test",]

write.csv(submission, file='rpartTheLot.csv', row.names=F, quote=F)
## Results



# Here is the model. We can see that if you were an adult male other than a noble then you

# likely died. Otherwise if three or more others in your group died then most likely you did too.

# Otherwise if you were in first or second class you likely survived. For the remainding third

# class passengers knowing the survival rate for others in your group was also influential.

fm



# And how well does our model predict the training set? It's about 85 per cent accurate on the

# training set. So definitely not over-predicting and better than the default of predicting

# everybody died at 62 per cent.

table(titanic$Survived, pred)