# Load packages

library('ggplot2') # visualization

library('ggthemes') # visualization

library('scales') # visualization

library('dplyr') # data manipulation

library('mice') # imputation

library('randomForest') # classification algorithm
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
# Finally, grab surname from passenger name

full$Surname <- sapply(full$Name,  

                      function(x) strsplit(x, split = '[,.]')[[1]][1])
cat(paste('We have <b>', nlevels(factor(full$Surname)), '</b> unique surnames. I would be interested to infer ethnicity based on surname --- another time.'))
# Create a family size variable including the passenger themselves

full$Fsize <- full$SibSp + full$Parch + 1



# Create a family variable 

full$Family <- paste(full$Surname, full$Fsize, sep='_')
# Use ggplot2 to visualize the relationship between family size & survival

ggplot(full[1:891,], aes(x = Fsize, fill = factor(Survived))) +

  geom_bar(stat='count', position='dodge') +

  scale_x_continuous(breaks=c(1:11)) +

  labs(x = 'Family Size') +

  theme_few()
# Discretize family size

full$FsizeD[full$Fsize == 1] <- 'singleton'

full$FsizeD[full$Fsize < 5 & full$Fsize > 1] <- 'small'

full$FsizeD[full$Fsize > 4] <- 'large'



# Show family size by survival using a mosaic plot

mosaicplot(table(full$FsizeD, full$Survived), main='Family Size by Survival', shade=TRUE)
# This variable appears to have a lot of missing values

full$Cabin[1:28]
# The first character is the deck. For example:

strsplit(full$Cabin[2], NULL)[[1]]
# Create a Deck variable. Get passenger deck A - F:

full$Deck<-factor(sapply(full$Cabin, function(x) strsplit(x, NULL)[[1]][1]))
# Passengers 62 and 830 are missing Embarkment

full[c(62, 830), 'Embarked']
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