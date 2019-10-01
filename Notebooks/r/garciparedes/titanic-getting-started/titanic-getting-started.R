rm(list = ls())
input.dir <- './../input/'
library(MASS)
removedColumnsIndex <- function(data, columIds) {

    setdiff(colnames(data), columIds)

}
columnsToNumeric <- function(data, columIds) {

    for (i in columIds) {

        data[,i] <- as.numeric(data[,i])

    }

    return(data)

}
regressionAge <- function(data) {

    columns.to.remove <- c('Name', 'Cabin', 'Ticket', 'PassengerId')

    data2 <- data[,removedColumnsIndex(data, columns.to.remove)]

    

    

    data.train <- data2[!is.na(data2[,'Age']),]    

    data.model <- lm( Age ~ . , data = data.train)

    data.test <- data2[is.na(data2[,'Age']),]

    

    data[is.na(data[,'Age']),'Age'] <- predict(data.model, data.test)

    return(data)

}
preprocessData <- function(data) {

    columns.to.remove <- c('Name', 'Cabin', 'Ticket')

    data.cleaned <- data[,removedColumnsIndex(data, columns.to.remove)]



    columns.to.numeric <- c('Sex', 'Embarked')

    data.cleaned <- columnsToNumeric(data.cleaned, columns.to.numeric)

    

    columns.to.scale <- setdiff(colnames(data.cleaned), c('PassengerId', 'Survived'))

    data.cleaned[,columns.to.scale] <- scale(data.cleaned[,columns.to.scale])

    data.cleaned[is.na(data.cleaned)] <- 0

    

    

    return(data.cleaned)

}
predictData <- function(model, data) {

    

    data.prediction <- cbind(data[,'PassengerId'] , as.numeric(predict(model, data)$class) - 1)

    colnames(data.prediction) <- c('PassengerId', 'Survived')



    return(data.prediction)

}
plotWithPCA <- function(data) {

    data.pca <- princomp(data[,removedColumnsIndex(data, 'Survived')])

    plot(data.pca$scores[,1:2], col = as.numeric(data[,'Survived']) + 2, pch = 20)

}
plotWithLDA <- function(obj, data) {

    plot(as.matrix(data[,removedColumnsIndex(data, 'Survived')]) %*% obj$scaling, 

         jitter(rep(0, dim(data)[1])), col = data[,'Survived'] + 2, 

         ylim = c(-0.05, 0.05), xlab = "", ylab = "", pch = 20)

}
data.train <- read.csv(paste(input.dir, 'train.csv', sep =""))

data.test <- read.csv(paste(input.dir, 'test.csv', sep =""))
summary(data.train)
head(data.train)
data.global <- regressionAge(rbind(data.train[,removedColumnsIndex(data, 'Survived')], data.test))

data.train[,'Age'] <- data.global[1:dim(data.train)[1], 'Age']

data.test[,'Age'] <- data.global[-c(1:dim(data.train)[1]), 'Age']
data.train.use <- preprocessData(data.train)

head(data.train.use)
pairs(data.train.use[, -c(1, 2)], col = as.numeric(data.train.use[, 2]) + 2, pch = 20)
plotWithPCA(data.train.use)
data.test.use <- preprocessData(data.test)

head(data.test.use)
data.classifier.formula <- as.factor(Survived) ~ .
data.classifier.lda <- lda(data.classifier.formula, data.train.use, prior = c(0.7, 0.3))



table(as.numeric(predict(data.classifier.lda, data.train.use)$class) - 1, data.train.use[,2])



mean(as.numeric(predict(data.classifier.lda, data.train.use)$class) - 1 == data.train.use[,2])
plotWithLDA(data.classifier.lda, data.train.use)
data.test.prediction.lda <- predictData(data.classifier.lda, data.test.use)



head(data.test.prediction.lda)



write.csv(data.test.prediction.lda, file = "prediction-lda.csv", row.names = FALSE)