# Load packages

library('ggplot2') # visualization

library('ggthemes') # visualization

library('scales') # visualization

library('dplyr') # data manipulation

library('mice') # imputation

library('randomForest') # classification algorith



train <- read.csv('../input/train.csv', stringsAsFactors = F)

test  <- read.csv('../input/test.csv', stringsAsFactors = F)



full  <- bind_rows(test, train) # bind training & test data



full

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
full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)