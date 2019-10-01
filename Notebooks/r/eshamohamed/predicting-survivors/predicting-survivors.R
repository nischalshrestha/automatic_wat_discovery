suppressMessages(library(Amelia)) 

suppressMessages(library(caret))

suppressMessages(library(pROC))

library(gridExtra)

library(corrplot)

library(plyr)
# load training data

t.train <- read.csv("../input/train.csv", stringsAsFactors = F)



# set all empty cells to NA

t.train <- sapply(t.train, function(x) {ifelse(x == "", NA, x)}, simplify = FALSE)

t.train <- as.data.frame(t.train, stringsAsFactors = F)



# do the same for test set

t.test <- read.csv("../input/test.csv", stringsAsFactors = F)

t.test <- sapply(t.test, function(x) {ifelse(x == "", NA, x)}, simplify = FALSE)

t.test <- as.data.frame(t.test, stringsAsFactors = F)



t.test$Survived <- NA

# combine train and test set for pre-processing

titanic <- rbind(t.train, t.test)



# get an idea of exisiting features

str(t.train)
missmap(titanic)
suppressMessages(attach(titanic))
table(Survived) 
p1 <- ggplot(as.data.frame(table(Survived, Pclass))) + aes(y=Freq,x=Pclass, fill=Survived) + 

            geom_col() + theme(legend.position="none")

p2 <- ggplot(as.data.frame(table(Survived, Sex))) + aes(y=Freq,x=Sex, fill=Survived) + 

            geom_col() + labs(y = "")

p3 <- ggplot(as.data.frame(table(Survived, SibSp))) + aes(y=Freq,x=SibSp, fill=Survived) + 

            geom_col() + theme(legend.position="none")

p4 <- ggplot(as.data.frame(table(Survived, Parch))) + aes(y=Freq,x=Parch, fill=Survived) + 

            geom_col() + labs(y = "")

grid.arrange(p1, p2, p3, p4, ncol=2, nrow=2)
Fsize <- SibSp + Parch + 1

FsizeD <- rep(NA, length(Fsize))

FsizeD[Fsize == 1] <- "Alone"

FsizeD[Fsize <= 3 & Fsize > 1] <- "Small"

FsizeD[Fsize <= 5 & Fsize > 3] <- "Medium"

FsizeD[Fsize > 5] <- "Large"

titanic$FsizeD <- FsizeD
suppressMessages(attach(titanic))

a1    <- which(is.na(Fare)) # there exists one entry with NA

titanic[a1,]
a2 <- which(titanic[,"Pclass"] ==3 & titanic[,"Embarked"] == "S")

a3 <- titanic[a2,]

head(a3) # just a check
ggplot(a3, aes(Fare)) + geom_histogram()
Impute.fare <- preProcess(as.data.frame(a3[,"Fare"]), method="medianImpute")

Impute.fare$median # value to be imputed
titanic[a1, "Fare"] <- Impute.fare$median

suppressMessages(attach(titanic))
sum(is.na(Pclass))

# nzchar is for checking is there are empty cells

a <- nzchar(Pclass); table(a)["FALSE"]

table(Pclass)



sum(is.na(Sex))

a <- nzchar(Sex); table(a)["FALSE"]

table(Sex)



sum(is.na(SibSp))

a <- nzchar(SibSp); table(a)["FALSE"]

table(SibSp)



sum(is.na(Parch))

a <- nzchar(Parch); table(a)["FALSE"]

table(Parch)



sum(is.na(Cabin))

a <- nzchar(Cabin); table(a)["FALSE"]

# 1014 out of 1309 cells are empty, just delet Cabin
sum(is.na(Embarked))

a   <- nzchar(Embarked) 

table(a)["FALSE"]
# There are two missing values. let's see what's uniques about them

a1 <- which(is.na(Embarked) == TRUE)

titanic[a1,]
a2 <- which(titanic[,"Pclass"] == 1 & titanic[,"Fare"] == 80)

a2
# Unfortunately not. Let's get first class passengers and see their distribution across Embarked



a3 <- which(titanic[,"Pclass"] == 1)

a4 <- titanic[a3,]

a5 <- is.na(a4[,"Embarked"])

a6 <- which(a5 == TRUE) # rows with NA

ggplot(a4[-a6,], aes(y=Fare, x=Embarked)) + geom_boxplot()
table(a4[-a6, "Embarked"])
impute.embarked        <- aggregate(Fare ~ Embarked, a4[-a6,], median)[1,1]

titanic[a1,"Embarked"] <- impute.embarked

colSums(apply(titanic, 2, is.na)) # just a check
# get out features that will not be used for imputation

notForImpute <- grep(paste(c("PassengerId", "Survived", "Name", "Ticket", 

                             "Cabin"), collapse = "|"), names(titanic))



# create a data frame from titanic with features that will be used for imputation 

imputeData  <- titanic[,-notForImpute]



# convert character string to factor

toFactor     <- grep(paste(c("Pclass", "Sex", "Embarked", "FsizeD"),

                           collapse = "|"), names(imputeData))

imputeData[toFactor] <- lapply(imputeData[toFactor], factor)
# preProcess function of the caret will ignor non-numeric features so create 

# dummy variables for factors

f               <- ~ Pclass + Sex + Embarked + FsizeD - 1

f               <- as.formula(f)

d2              <- dummyVars( f, data = imputeData, levelsOnly = TRUE)

d2              <- predict(d2, imputeData)

imputeDataDummy <- cbind(imputeData[,-toFactor], d2)
# do knn imputation with different values of nearest neighbours, k

kNumber         <- 2:4

theDataImpute   <- matrix(NA, nrow(imputeDataDummy), length(kNumber)) # to store predicted Age



for(i in 1:length(kNumber)){

  preProcAge        <- preProcess(imputeDataDummy, method = "knnImpute", k = kNumber[i] )

  theDataImpute[,i] <- predict(preProcAge, imputeDataDummy)$Age

}
AgeNorm <- preProcess(as.data.frame(titanic$Age), method = c("center", "scale"))

AgeNorm <- predict(AgeNorm, as.data.frame(titanic$Age))

AgeNorm <- unlist(AgeNorm)
ggplot(as.data.frame(AgeNorm), aes(AgeNorm)) + 

  geom_histogram(binwidth=.4, colour="black", fill="white", aes(y = ..density..)) +

  geom_density(data = as.data.frame(AgeNorm), na.rm = TRUE) +

  geom_density(aes(theDataImpute[,1]), col = "red") +  # k = 2

  geom_density(aes(theDataImpute[,2]), col = "blue") + # k = 3

  geom_density(aes(theDataImpute[,3]), col = "green")  # k = 4
titanic$Age <- theDataImpute[,1]
# remove columns with many categories or without "useful" feature 

toRm      <- grep(paste(c("PassengerId","Survived","Name", "Ticket","Cabin"), 

                   collapse = "|"), names(titanic))

checkCorr <- titanic[,-toRm]



checkCorr$Sex       <- revalue(checkCorr$Sex, c("male" = 1, "female" = 2))

checkCorr$Embarked  <- revalue(checkCorr$Embarked, c("S" = 1, "Q" = 2,  "C" = 3))

checkCorr$FsizeD    <- revalue(checkCorr$FsizeD, c("Alone" = 1, "Small" = 2, 

                                                   "Medium" = 3, "Large" = 4))



# convert characters to numeric

checkCorr$Sex       <- as.numeric(checkCorr$Sex)

checkCorr$Embarked  <- as.numeric(checkCorr$Embarked)

checkCorr$FsizeD    <- as.numeric(checkCorr$FsizeD)



corrData            <- cor(checkCorr) # find pairwise correlation



# plot correlation matrix

corrplot(corrData, method = "square", order = "hclust")
# identify correlated predictors with correlation > 0.8 for removal

find.corr <- findCorrelation(corrData, cutoff = .8, names = FALSE, exact = TRUE)

find.corr # all pairwise correlation is less than .8 so we proceed to model training
str(titanic)
# remove features that will not be used for model training leaving PassengerId

toRm      <- grep(paste(c("Name", "Ticket", "Cabin"), collapse = "|"), names(titanic))

titanic   <- titanic[,-toRm]



# convert characters to factors

titanic$Pclass     <- factor(titanic$Pclass)

titanic$Sex        <- factor(titanic$Sex)

titanic$Embarked   <- factor(titanic$Embarked)

titanic$FsizeD     <- factor(titanic$FsizeD)
# separate training from test set

titanic.train          <- titanic[!is.na(titanic$Survived),]

titanic.train$Survived <- factor(ifelse(titanic.train$Survived == 1, "yes", "no"))

titanic.test           <- titanic[is.na(titanic$Survived),]

titanic.test           <- titanic.test[,-grep("Survived", names(titanic.test))]

cntr <- trainControl(method = "repeatedcv", repeats = 5)



f    <- Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + FsizeD 

f    <- as.formula(f)
set.seed(009)

svm.fit <- train(f, data = titanic.train,

                 method = "svmRadial",

                 trControl = cntr,

                 preProcess = c("center", "scale"),

                 verbose = FALSE)



set.seed(009)

rf.fit <- train(f, data = titanic.train,

                method = "rf",

                trControl = cntr,

                preProcess = c("center", "scale"),

                verbose = FALSE)



set.seed(009)

gbm.fit <- train(f, data = titanic.train,

                 method = "gbm",

                 trControl = cntr,

                 preProcess = c("center", "scale"),

                 verbose = FALSE)



set.seed(009)

log.fit <- train(f, data = titanic.train,

                 method = "glm",

                 family = binomial(link = "logit"),

                 trControl = cntr,

                 preProcess = c("center", "scale"))

# put them together for comparison

out <- resamples(list(SVM = svm.fit, RF = rf.fit, GBM = gbm.fit, Log = log.fit))

summary(out)
bwplot(out)
dotplot(out)
# There does not seem to be much differences especially between GBM, RF and SVM. 

# Let us do a hypothesis test
# Hypothesis test: H_0 is such that there is no difference in performance 

# between the models trained

modelDiff <- diff(out)

summary(modelDiff)
print(gbm.fit)
# confusion matrix

confusionMatrix(gbm.fit)
# variable importance

plot(varImp(gbm.fit))
gbm.predict <- predict(gbm.fit, newdata = titanic.test)

gbm.predict <- ifelse(gbm.predict == "no", 0, 1)



Out <- as.data.frame(titanic.test$PassengerId)

Out$Survived <- gbm.predict

colnames(Out)[1] <- "PassengerId"



write.csv(Out, file = "TitanicSurvived.csv", row.names = F)


