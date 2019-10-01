library('ggplot2') # visualization

library('ggthemes') # visualization

library('scales') # visualization

library('dplyr') # data manipulation

library('mice') # imputation

library('randomForest') # classification algorithm



train <- read.csv('../input/train.csv', stringsAsFactors = F) #training data

test  <- read.csv('../input/test.csv', stringsAsFactors = F) #testing data



full  <- bind_rows(train, test) # bind training & test data



# check data

str(train)

str(test)



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



Splitter <- function(x){

	strsplit(x, split = '[,.]')[[1]][1]

}



full$Surname <- sapply(full$Name,  Splitter)

                          

full$Name

full$Surname

                          
