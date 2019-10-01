library(readr)

library(magrittr)



Train <- read_csv("../input/train.csv")

Test <- read_csv("../input/test.csv")

Train$Pclass %<>% as.character()

Test$Pclass %<>% as.character()
library(aod)



mylogit <- glm(Survived ~ Pclass + Sex + Fare + SibSp + Parch,

               family = binomial("logit"), data = Train)

Survived <- predict(mylogit, Test[, c(2,4,9,6,7)])

Survived <- ifelse(Survived >= 0, 1, 0)

Res <- data.frame(Test[,1], Survived)

write.csv(Res, 'titanic_pred.csv', row.names = F)