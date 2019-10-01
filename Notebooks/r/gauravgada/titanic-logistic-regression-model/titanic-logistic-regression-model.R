library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function

library(AUC)



train = read.csv("../input/train.csv")

test = read.csv("../input/test.csv")





attach(train)

train$Age[is.na(Age)] = mean(train$Age, na.rm = T)

test$Age[is.na(test$Age)] = mean(test$Age, na.rm = T)



mod <- glm(Survived ~ Pclass + Sex + Age + SibSp,

           data=train,

           family="binomial")



fits = fitted(mod)

rr = roc(fits, factor(train$Survived))

auc(rr)

plot(rr)



predicted = predict(mod, test, type = "response")



test$p[predicted >= 0.53] = 1

test$p[predicted < 0.53] = 0



# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)

solution <- data.frame(PassengerId = test$PassengerId, Survived = test$p)



# Write the solution to file

write.csv(solution, file = 'logistic_regression.csv', row.names = F)
