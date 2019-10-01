# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



# Load packages

library('ggplot2') # visualization

library('ggthemes') # visualization

library('scales') # visualization

library('dplyr') # data manipulation

library('mice') # imputation

library('randomForest') # classification algorithm



# Any results you write to the current directory are saved as output.
train <- read.csv('../input/train.csv', stringsAsFactors = F)

test  <- read.csv('../input/test.csv', stringsAsFactors = F)



full  <- bind_rows(train, test) # bind training & test data



# check data

str(full)
# Grab title from passenger names

full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)



# Show title counts by sex

table(full$Sex, full$Title)
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
# Create a family size variable including the passenger themselves

full$Fsize <- full$SibSp + full$Parch + 1



# Create a family variable 

full$Family <- paste(full$Surname, full$Fsize, sep='_')



# Use ggplot2 to visualize the relationship between family size & survival

ggplot(full[1:891,], aes(x = Pclass, fill = factor(Survived))) +

  geom_bar(stat='count', position='dodge') +

  scale_x_continuous(breaks=c(1:90)) +

  labs(x = 'Class') +

  theme_few()
# Discretize family size

full$FsizeD[full$Fsize == 1] <- 'singleton'

full$FsizeD[full$Fsize < 5 & full$Fsize > 1] <- 'small'

full$FsizeD[full$Fsize > 5] <- 'large'



# Show family size by survival using a mosaic plot

mosaicplot(table(full$FsizeD, full$Survived), main='Family Size by Survival', shade=TRUE)
str(full)
factor_vars <- c('PassengerId','Pclass','Sex','Embarked',

                 'Title','Family','FsizeD')



full[factor_vars] <- lapply(full[factor_vars], function(x) as.factor(x))



# Set a random seed

set.seed(129)



# Perform mice imputation, excluding certain less-than-useful variables:

mice_mod <- mice(full[, !names(full) %in% c('PassengerId','Name','Ticket','Cabin','Family','Surname','Survived')], method='rf') 
mice_output <- complete(mice_mod)
par(mfrow=c(1,2))

hist(full$Age, freq=F, main='Age: Original Data', 

  col='darkgreen', ylim=c(0,0.04))

hist(mice_output$Age, freq=F, main='Age: MICE Output', 

  col='lightgreen', ylim=c(0,0.04))
full$Age <- mice_output$Age



sum(is.na(full$Age))
ggplot(full[1:891,], aes(Age, fill = factor(Survived))) + 

  geom_histogram() + 

  # I include Sex since we know (a priori) it's a significant predictor

  facet_grid(.~Sex) + 

  theme_few()
full$Child[full$Age < 18] <- 'Child'

full$Child[full$Age >= 18] <- 'Adult'



# Show counts

table(full$Child, full$Survived)
full$Mother <- 'Not Mother'

full$Mother[full$Sex == 'female' & full$Parch > 0 & full$Age > 18 & full$Title != 'Miss'] <- 'Mother'



# Show counts

table(full$Mother, full$Survived)
full$Child  <- factor(full$Child)

full$Mother <- factor(full$Mother)
md.pattern(full)
train <- full[1:891,]

test <- full[892:1309,]
# Set a random seed

set.seed(754)



# Build the model (note: not all possible variables are used)

rf_model <- randomForest(data = train,factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + Child + Mother)



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
# Predict using the test set

test$Survived <- predict(rf_model, test)



# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)

solution <- data.frame(PassengerID = test$PassengerId, Survived = test$Survived)



# Write the solution to file

write.csv(solution, file = 'rf_mod_Solution.csv', row.names = F)