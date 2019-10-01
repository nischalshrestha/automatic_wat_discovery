library(FFTrees)

library(party)

library(randomForest)

library(gbm)

library(dplyr)

library(intubate)
titanic <- read.csv("../input/train.csv", stringsAsFactors = FALSE)

str(titanic)
names(titanic) <- tolower(names(titanic))

titanic$sex <- as.factor(titanic$sex)

titanic$embarked <- as.factor(titanic$embarked)
ggplot2::ggplot(titanic, ggplot2::aes(age)) +

    ggplot2::geom_density(fill = "blue", alpha = .6)
mean(is.na(titanic$age))
age_prediction <- lm(age ~ survived + pclass + fare, data = titanic)

summary(age_prediction)
titanic$age[is.na(titanic$age)] <- predict(age_prediction,

    newdata = titanic[is.na(titanic$age),])



# Check NAs in age

sum(is.na(titanic$age))
logi <- titanic %>% 

    select(survived, pclass, sex, age, sibsp) %>% 

    ntbt_glm(survived ~ ., family = binomial)



summary(logi)
logi_pred <- predict(logi, type = "response")

survivors_logi <- rep(0, nrow(titanic))

survivors_logi[logi_pred > .5] <- 1

table(model = survivors_logi, real = titanic$survived)
test <- read.csv("../input/test.csv", 

    stringsAsFactors = FALSE, 

    na.strings = "")



names(test) <- tolower(names(test))

test$sex <- as.factor(test$sex)



test_logi_pred <- predict(logi, test, type = "response")

surv_test_logi <- data.frame(PassengerId = test$passengerid, 

    Survived = rep(0, nrow(test)))



surv_test_logi$Survived[test_logi_pred > .5] <- 1

table(surv_test_logi$Survived)
fftitanic <- titanic %>% 

    select(age, pclass, sex, sibsp, fare, survived) %>% 

    ntbt(FFTrees, survived ~ .)



# Plotting of the best tree

plot(fftitanic, 

     main = "Titanic", 

     decision.names = c("Not Survived", "Survived"))
ffpred <- ifelse(test$sex != "male", 1,

                 ifelse(test$pclass > 2, 0,

                        ifelse(test$fare < 26.96, 0,

                               ifelse(test$age >= 21.36, 0, 1))))
# FFTree doesn't deal with NAs, I assign a 0 to them

ffpred[is.na(ffpred)] <- 0

ffpred <- data.frame(PassengerId = test$passengerid, Survived = ffpred)
partyTitanic <- titanic %>% 

    select(age, pclass, sex, sibsp, fare, survived) %>% 

    ntbt(ctree, as.factor(survived) ~ .)



# Plot the resulting tree

plot(partyTitanic, main = "Titanic prediction", type = "simple",

     inner_panel = node_inner(partyTitanic, 

                              pval = FALSE),

     terminal_panel = node_terminal(partyTitanic,

                                    abbreviate = TRUE,

                                    digits = 1,

                                    fill = "white"))
train_party <- Predict(partyTitanic)

table(tree = train_party, real = titanic$survived)
party_pred <- Predict(partyTitanic, newdata = test)

party_pred <- as.numeric(party_pred) - 1

party_pred <- data.frame(PassengerId = test$passengerid, 

                         Survived = party_pred)
set.seed(123)



# Bagging model building

titanic_bag <- titanic %>% 

    select(survived, age, pclass, sex, sibsp, fare, parch) %>% 

    ntbt_randomForest(as.factor(survived) ~ ., mtry = 6)



# Bagging and Random Forest don't deal with NAs

test$age[is.na(test$age)] <- median(test$age, na.rm = TRUE)



# The usual test set prediction

bag_pred <- predict(titanic_bag, test)



# Check if there are NAs in prediction and substitute them

sum(is.na(bag_pred))
bag_pred[is.na(bag_pred)] <- 1

bag_pred <- data.frame(PassengerId = test$passengerid, 

                       Survived = bag_pred, 

                       row.names = 1:length(bag_pred))
set.seed(456)



# Random Forest model building

titanic_rf <- titanic %>% 

    select(survived, age, pclass, sex, sibsp, fare, parch) %>% 

    ntbt_randomForest(as.factor(survived) ~ ., mtry = 3, ntree = 5000)



# Prediction

rf_pred <- predict(titanic_rf, test)

rf_pred[is.na(rf_pred)] <- 1

rf_pred <- data.frame(PassengerId = test$passengerid, Survived = rf_pred, row.names = 1:nrow(test))



plot(titanic_rf)
set.seed(415)



# Use the cforest function from party package

titanic_rf_party <- titanic %>% 

    select(survived, age, pclass, sex, sibsp, fare, parch) %>% 

    ntbt(cforest, as.factor(survived) ~ ., 

            controls = cforest_unbiased(ntree = 5000, mtry = 3))



# Prediction of the test set

rf_party_pred <- predict(titanic_rf_party, 

                         test, 

                         OOB = TRUE, 

                         type = "response")

rf_party_pred <- data.frame(PassengerId = test$passengerid, 

                            Survived = rf_party_pred)
set.seed(999)



# Boosting model building

titanic_boost <- titanic %>% 

    select(survived, age, pclass, sex, sibsp, fare, parch) %>% 

    ntbt(gbm, survived ~ .,

         distribution = "bernoulli",

         n.trees = 5000,

         interaction.depth = 3)



# Boosting prediction

boost_pred <- predict(titanic_boost, test, n.trees = 5000, type = "response")

test_boost <- rep(0, nrow(test))

test_boost[boost_pred >= .5] <- 1

test_boost <- data.frame(PassengerId = test$passengerid,

                         Survived = test_boost)
