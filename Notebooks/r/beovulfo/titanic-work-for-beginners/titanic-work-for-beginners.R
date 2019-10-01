# Load packages

library('ggplot2') # visualization

library('ggthemes') # visualization

library('scales') # visualization

library('mice') # imputation

library('randomForest') # classification algorithm

library('caret')

library('Amelia')

library('dplyr') # data manipulation
training.data.raw <- read.csv('../input/train.csv',header=T,na.strings=c(""),stringsAsFactors = F)

test.data.raw <- read.csv('../input/test.csv',header=T,na.strings=c(""),stringsAsFactors = F)

full  <- bind_rows(training.data.raw, test.data.raw) # bind training & test data
#Print a summary of NA values on data by column

sapply(full,function(x) sum(is.na(x)))

#sapply(full, function(x) length(unique(x)))
#install.packages("Amelia")

#install.packages("pscl")

#install.packages("lattice")
missmap(full, main = "Missing values vs observed")
head(full)
summary(full)
# Grab title from passenger names

full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)



# Show title counts by sex

table(full$Pclass, full$Title)
# Finally, grab surname from passenger name

full$Surname <- sapply(full$Name,  

                      function(x) strsplit(x, split = '[,.]')[[1]][1])
# Create a family size variable including the passenger themselves

full$Fsize <- full$SibSp + full$Parch + 1

# Discretize family size

full$FsizeD[full$Fsize == 1] <- 'singleton'

full$FsizeD[full$Fsize < 5 & full$Fsize > 1] <- 'small'

full$FsizeD[full$Fsize > 4] <- 'large'



# Show family size by survival using a mosaic plot

mosaicplot(table(full$FsizeD, full$Survived), main='Family Size by Survival', shade=TRUE)

# Create a family variable 

#full$Family <- paste(full$Surname, full$Fsize, sep='_')
full[full$PassengerId[is.na(full$Embarked)],]
# Get rid of our missing passenger IDs

embark_fare <- full %>%

  filter(PassengerId != 62 & PassengerId != 830)



# Use ggplot2 to visualize embarkment, passenger class, & median fare

ggplot(embark_fare, aes(x = Embarked, y = Fare, fill = factor(Pclass))) +

  geom_boxplot() +

  geom_hline(aes(yintercept=80), 

    colour='red', linetype='dashed', lwd=2) +

  scale_y_continuous(labels=dollar_format()) +

  theme_few()
# Since their fare was $80 for 1st class, they most likely embarked from 'C'

full$Embarked[c(62, 830)] <- 'C'
full[full$PassengerId[is.na(full$Fare)],]
mean_mv <- mean(na.omit(full$Fare[full$Embarked=='S' & full$Pclass==3]))

full$Fare[1044] <- mean_mv
ggplot(full[full$Pclass == '3' & full$Embarked == 'S', ], 

  aes(x = Fare)) +

  geom_density(fill = '#99d6ff', alpha=0.4) + 

  geom_vline(aes(xintercept=median(Fare, na.rm=T)),

    colour='red', linetype='dashed', lwd=1) +

  scale_x_continuous(labels=dollar_format()) +

  theme_few()
# Titles with very low cell counts to be combined to "rare" level

rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 

                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')



# Also reassign mlle, ms, and mme accordingly

full$Title[full$Title == 'Mlle']        <- 'Miss' 

full$Title[full$Title == 'Ms']          <- 'Miss'

full$Title[full$Title == 'Mme']         <- 'Mrs' 

full$Title[full$Title %in% rare_title]  <- 'Rare Title'



# Show title counts by sex again

table(full$Sex, full$Title)
colnames(full)
# Make variables factors into factors

factor_vars <- c('Pclass','Embarked','SibSp','Parch',

                 'Title')



full[factor_vars] <- lapply(full[factor_vars], function(x) as.factor(x))



# Set a random seed

set.seed(129)



# Perform mice imputation, excluding certain less-than-useful variables:

mice_mod <- mice(full[, !names(full) %in% c('PassengerId','Name','Ticket','Cabin','Family','Surname','Survived')], method='rf') 
# Save the complete output 

mice_output <- complete(mice_mod)

# Plot age distributions

par(mfrow=c(1,2))

hist(full$Age, freq=F, main='Age: Original Data', 

  col='darkgreen', ylim=c(0,0.04))

hist(mice_output$Age, freq=F, main='Age: MICE Output', 

  col='lightgreen', ylim=c(0,0.04))
# Replace Age variable from the mice model.

full$Age <- mice_output$Age



# Show new number of missing Age values

sum(is.na(full$Age))
# First we'll look at the relationship between age & survival

ggplot(full[1:891,], aes(Age, fill = factor(Survived))) + 

  geom_histogram() + 

  # I include Sex since we know (a priori) it's a significant predictor

  facet_grid(.~Sex) + 

  theme_few()
# Create the column child, and indicate whether child or adult

full$Child[full$Age < 18] <- 'Child'

full$Child[full$Age >= 18] <- 'Adult'



# Show counts

table(full$Child, full$Survived)
# Finish by factorizing our two new factor variables

full$Child  <- factor(full$Child)

full$Sex <- factor(full$Sex)
# Split the data back into a train set and a test set

train <- full[1:891,]

test <- full[892:1309,]
is.factor(train$Sex)

is.factor(train$Embarked)
contrasts(train$Sex)

#contrasts(data$Embarked)
sapply(train,function(x) sum(is.na(x)))
set.seed(754)



# Build the model (note: not all possible variables are used)

rf_model <- randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + 

                                            Fare + Embarked + 

                                             Child ,

                                            data = train)
# Show model error

plot(rf_model, ylim=c(0,0.36))

legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)
# Get importance

importance    <- importance(rf_model)

varImportance <- data.frame(Variables = row.names(importance), 

                            Importance = round(importance[ ,'MeanDecreaseGini'],2))



# Create a rank variable based on importance

rankImportance <- varImportance %>%

  mutate(Rank = paste0('#',dense_rank(desc(Importance))))



# Use ggplot2 to visualize the relative importance of variables

ggplot(rankImportance, aes(x = reorder(Variables, Importance), 

    y = Importance, fill = Importance)) +

  geom_bar(stat='identity') + 

  geom_text(aes(x = Variables, y = 0.5, label = Rank),

    hjust=0, vjust=0.55, size = 4, colour = 'red') +

  labs(x = 'Variables') +

  coord_flip() + 

  theme_few()
# model <- glm(Survived ~.,family=binomial(link='logit'),data=train)
summary(rf_model)
# Predict using the test set

prediction <- predict(rf_model, test)



# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)

solution <- data.frame(PassengerID = test$PassengerId, Survived = prediction)



# Write the solution to file

write.csv(solution, file = 'rf_mod_Solution3.csv', row.names = F)
anova(rf_model, test="Chisq")
library(pscl)

pR2(model)
fitted.results <- predict(model,newdata=subset(test,select=c(2,3,4,5,6,7,8)),type='response')

fitted.results <- ifelse(fitted.results > 0.5,1,0)

misClasificError <- mean(fitted.results != test$Survived)

print(paste('Accuracy',1-misClasificError))
install.packages("ROCR")
library(ROCR)

p <- predict(model, newdata=subset(test,select=c(2,3,4,5,6,7,8)), type="response")

pr <- prediction(p, test$Survived)

prf <- performance(pr, measure = "tpr", x.measure = "fpr")

plot(prf)



auc <- performance(pr, measure = "auc")

auc <- auc@y.values[[1]]

auc
test.raw <- read.csv('titanic/test.csv',header=T,na.strings=c(""))
sapply(test.raw,function(x) sum(is.na(x)))

sapply(test.raw, function(x) length(unique(x)))
missmap(test.raw, main = "Missing values vs observed")
vnames <- colnames(test.raw)

vnames
vnames <- vnames[c(2,4,5,7)]

vnames
test <- subset(test.raw,select=vnames)
test$Age[is.na(test$Age)] <- median(data$Age,na.rm=T)

test$Fare[is.na(test$Fare)] <- mean_fare_NAN
# Get rid of our missing passenger IDs

embark_fare <- test 

#full %>%

  #filter(PassengerId != 62 & PassengerId != 830)



# Use ggplot2 to visualize embarkment, passenger class, & median fare

ggplot(embark_fare, aes(x = Embarked, y = Fare, fill = factor(Pclass))) +

  geom_boxplot() +

  geom_hline(aes(yintercept=80), 

    colour='red', linetype='dashed', lwd=2) +

  scale_y_continuous(labels=dollar_format()) +

  theme_few()
train <- data[1:889,]
model <- glm(Survived ~.,family=binomial(link='logit'),data=train)
summary(model)
anova(model, test="Chisq")
library(pscl)

pR2(model)
#

p <- predict(model, newdata=subset(test,select=vnames), type="response")
test[153,]
# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)

solution <- data.frame(PassengerID = test.raw$PassengerId, Survived = round(p))



# Write the solution to file

write.csv(solution, file = 'rf_mod_Solution2.csv', row.names = F)
