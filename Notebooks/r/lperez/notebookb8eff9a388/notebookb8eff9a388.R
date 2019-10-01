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

full$Title <- gsub('(.*, )|(\\.\\s.*)', '', full$Name)



# Show title counts by sex

table(full$Sex, full$Title)
full[c("Title","Name")]
rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 

                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')



full$Title[full$Title == 'Mlle'] <- 'Miss'

full$Title[full$Title == 'Ms'] <- 'Miss'

full$Title[full$Title == 'Mme'] <- 'Miss'

full$Title[full$Title %in% rare_title] <- 'Rare Title'

table(full$Sex, full$Title)

# Finally, grab surname from passenger name

full$Surname <- sapply(full$Name,  

                      function(x) strsplit(x, split = ',')[[1]][1])



cat(paste('We have <b>', nlevels(factor(full$Surname)), '</b> unique surnames. I would be interested to infer ethnicity based on surname --- another time.'))
# Create a family size variable including the passenger themselves

full$Fsize <- full$SibSp + full$Parch + 1



# Create a family variable 

full$Family <- paste(full$Surname, full$Fsize, sep='_')
full[c('Family')]
newdata <- subset(full, Sex == 'male')
options(repr.plot.height=3)

ggplot(subset(full[1:891,], Sex == 'female'), aes(x = Fsize, fill = factor(Survived))) +

  geom_bar(stat='count', position='dodge') +

  scale_x_continuous(breaks=c(1:11)) +

  labs(x = 'Family Size') +

  theme_few()
ggplot(subset(full[1:891,], Sex == 'female'))