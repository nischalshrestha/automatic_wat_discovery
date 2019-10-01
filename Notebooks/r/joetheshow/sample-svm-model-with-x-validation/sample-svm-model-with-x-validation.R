# This R script will run on our backend. You can write arbitrary code here!

#loading packages
library(impute)
#library(DMwR) #Kaggle does not support DMwR?)
library(e1071)

#Load input data, treat empty data cells as NAs
train.raw <- read.csv("../input/train.csv", na.strings = "")
test.raw  <- read.csv("../input/test.csv", na.strings = "")

#Check for missing values
sapply(train.raw, function(x) {any(is.na(x))})
sapply(test.raw, function(x) {any(is.na(x))})
#in the training data, only the "Age" column has NAs, but in the test data, both Age and Fare has NAs

#Quick data-filling with KNN (Kaggle does not support DMwR?)
#train.clean = knnImputation(train.raw, k = 5, scale = T, meth = "weightAvg")
#test.clean = knnImputation(test.raw, k = 5, scale = T, meth = "weightAvg")
train.clean = train.raw
test.clean = test.raw

#Create 3 SVM models and perform cross-validations
formula = Survived ~ Sex + Pclass

rows = nrow(train.clean)
#arbitrary 10 folds for cross-validation
k = 10 
set.seed(999)
score = 0

#training svm
for (i in 1:k) {
    valIndices = sample(1:rows, rows/k)
    model.svm = svm(formula, data = train.clean[-valIndices, ], kernel="radial")
    val.pred = predict(model.svm, train.clean[valIndices, ])
    val.pred = ifelse(val.pred > 0.5, 1, 0)
    result = table(predict = val.pred, actual = train.clean$Survived[valIndices])
    newScore = (result[1,1] + result[2,2]) / sum(result)
    print(paste0("cross validation set ", k, " score = ", newScore))
    score = score + newScore
}
score = score / k
print(paste("Final Score = ", score))
