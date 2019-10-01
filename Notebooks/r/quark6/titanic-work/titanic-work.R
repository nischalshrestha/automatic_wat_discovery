library('ggplot2') # visualization

library('ggthemes') # visualization

library('scales') # visualization

library('dplyr') # data manipulation

library('mice') # imputation

library('randomForest')



# The train and test data is stored in the ../input directory

train <- read.csv("../input/train.csv", stringsAsFactors = F)

test  <- read.csv("../input/test.csv", stringsAsFactors = F)

combined  <- bind_rows(train, test) # bind training & test data

# We can inspect the train data. The results of this are printed in the log tab below

#-----------------------------------------

str(combined)
combined[combined==""]<-NA
sum(is.na(combined$Sex))
#amount of male compared to female

ggplot(combined[1:891,], aes(x = Sex, fill = factor(Survived))) +

  geom_bar(stat='count', position='dodge') +

  scale_x_discrete(breaks=c('female','male')) +

  labs(x = 'Sex') +

  theme_few()
sum(is.na(combined$Embarked))
filter(combined, is.na(Embarked))
# Get rid of our missing passenger IDs

embark_fare <- combined %>%

  filter(PassengerId != 62 & PassengerId != 830)



# Use ggplot2 to visualize embarkment, passenger class, & median fare

ggplot(embark_fare, aes(x = Embarked, y = Fare, fill = factor(Pclass))) +

  geom_boxplot() +

  geom_hline(aes(yintercept=80), 

    colour='red', linetype='dashed', lwd=2) +

  scale_y_continuous(labels=dollar_format()) +

  theme_few()
# Since their fare was $80 for 1st class, they most likely embarked from 'C'

combined$Embarked[c(62, 830)] <- 'C'
#amount of male compared to female

ggplot(combined[1:891,], aes(x = Embarked, fill = factor(Survived))) +

  geom_bar(stat='count', position='dodge') +

  scale_x_discrete(breaks=c('C','Q','S')) +

  labs(x = 'Embarked') +

  theme_few()
sum(is.na(combined$Pclass))
#amount of male compared to female

ggplot(combined[1:891,], aes(x = Pclass, fill = factor(Survived))) +

  geom_bar(stat='count', position='dodge') +

  scale_x_continuous(breaks=c(1:3)) +

  labs(x = 'Passenger Class') +

  theme_few()
# Create a family size variable

combined$FamSize <- combined$SibSp + combined$Parch + 1
# Use ggplot2 to visualize the relationship between family size & survival

ggplot(combined[1:891,], aes(x = FamSize, fill = factor(Survived))) +

  geom_bar(stat='count', position='dodge') +

  scale_x_continuous(breaks=c(1:11)) +

  labs(x = 'Family Size') +

  theme_few()
sum(is.na(combined$Fare))
filter(combined, is.na(Fare))
ggplot(combined[combined$Pclass == '3' & combined$Embarked == 'S', ], 

  aes(x = Fare)) +

  geom_density(fill = '#99d6ff', alpha=0.4) + 

  geom_vline(aes(xintercept=median(Fare, na.rm=T)),

    colour='red', linetype='dashed', lwd=1) +

  scale_x_continuous(labels=dollar_format()) +

  theme_few()
median(combined[combined$Pclass == '3' & combined$Embarked == 'S', ]$Fare, na.rm = TRUE)
# Replace missing fare value with median fare for class/embarkment

combined$Fare[1044] <- median(combined[combined$Pclass == '3' & combined$Embarked == 'S', ]$Fare, na.rm = TRUE)
# Use ggplot2 to visualize the relationship between fare & survival

# First we'll look at the relationship between age & survival

ggplot(combined[1:891,], aes(Fare, fill = factor(Survived))) + 

  geom_histogram(binwidth = 6) + 

  facet_grid(.~Sex) + 

  theme_few()
sum(is.na(combined$Age))
# Grab title from passenger names

combined$Title <- gsub('(.*, )|(\\..*)', '', combined$Name)



# Titles with very low cell counts to be combined to "rare" level

rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 

                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')



# Also reassign mlle, ms, and mme accordingly

combined$Title[combined$Title == 'Mlle']        <- 'Miss' 

combined$Title[combined$Title == 'Ms']          <- 'Miss'

combined$Title[combined$Title == 'Mme']         <- 'Mrs' 

combined$Title[combined$Title %in% rare_title]  <- 'Rare_Title'



# Show title counts by sex again

table(combined$Sex, combined$Title)
# Finally, grab surname from passenger name



combined$Surname <- sapply(combined$Name,  

                      function(x) strsplit(x, split = '[,.]')[[1]][1])
# Create a family variable 

combined$Fam <- paste(combined$Surname, combined$FamSize, sep='_')
# Discretize family size

combined$Fam_SizeD[combined$FamSize == 1] <- 'single'

combined$Fam_SizeD[combined$FamSize < 5 & combined$FamSize > 1] <- 'small'

combined$Fam_SizeD[combined$FamSize > 4] <- 'large'
# Make variables factors into factors

factor_vars <- c('PassengerId','Pclass','Sex','Embarked',

                 'Title','Surname','Fam','Fam_SizeD')



combined[factor_vars] <- lapply(combined[factor_vars], function(x) as.factor(x))



# Set a random seed

set.seed(129)



# Perform mice imputation, excluding certain less-than-useful variables:

mice_mod <- mice(combined[, !names(combined) %in% c('PassengerId','Name','Ticket','Cabin','Fam','Surname','Survived')], method='rf') 
# Save the complete output 

mice_output <- complete(mice_mod)
# Plot age distributions

par(mfrow=c(1,2))

hist(combined$Age, freq=F, main='Age: Original Data', 

  col='darkgreen', ylim=c(0,0.04))

hist(mice_output$Age, freq=F, main='Age: MICE Output', 

  col='lightgreen', ylim=c(0,0.04))
# Replace Age variable from the mice model.

combined$Age <- mice_output$Age



# Show new number of missing Age values

sum(is.na(combined$Age))
# First we'll look at the relationship between age & survival

ggplot(combined[1:891,], aes(Age, fill = factor(Survived))) + 

  geom_histogram(binwidth=2) + 

  facet_grid(.~Sex) + 

  theme_few()
# Split the data back into a train set and a test set

train <- combined[1:891,]

test <- combined[892:1309,]
# Create the column child, and indicate whether child or adult

combined$Status[combined$Age < 18] <- 'Child'

combined$Status[combined$Age >= 18] <- 'Adult'



# Show counts

table(combined$Status, combined$Survived)
# Adding Mother variable

combined$Mother <- 'Not Mother'

combined$Mother[combined$Sex == 'female' & combined$Parch > 0 & combined$Age > 18 & combined$Title != 'Miss'] <- 'Mother'



# Show counts

table(combined$Mother, combined$Survived)
# Finish by factorizing our two new factor variables

combined$Status  <- factor(combined$Status)

combined$Mother <- factor(combined$Mother)
md.pattern(combined)
# Set a random seed

set.seed(1000)



# Build the model (note: not all possible variables are used)

RanFor_model <- randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + 

                                            Fare + Embarked + Title + 

                                            Fam_SizeD,

                                            data = train,

                                            ntree = 900,

                                            mtry = 6,

                                            nodesize = 0.01 * nrow(test))



# Show model error

plot(RanFor_model, ylim=c(0,0.36))

legend('bottomleft',colnames(RanFor_model$err.rate), col=1:3, fill=1:3, bty='n', lty=1:3, cex=0.8)
# Get importance

importance <- importance(RanFor_model)

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

prediction <- predict(RanFor_model, test)



# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)

solution <- data.frame(PassengerID = test$PassengerId, Survived = prediction)



# Write the solution to file

write.csv(solution, file = 'RanFor_Solution.csv', row.names = F)