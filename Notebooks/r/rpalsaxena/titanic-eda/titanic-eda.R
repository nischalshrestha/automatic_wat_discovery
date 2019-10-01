library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function

library('dplyr')

train <- read.csv('../input/train.csv' , strip.white = TRUE, stringsAsFactors = F)

test <- read.csv('../input/test.csv' , strip.white = TRUE, stringsAsFactors = F)



full <- bind_rows(train, test)

str(full)
head(full, 1)
full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)

table(full$Sex, full$Title)
rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 

                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')



# Also reassign mlle, ms, and mme accordingly

full$Title[full$Title == 'Mlle']        <- 'Miss' 

full$Title[full$Title == 'Ms']          <- 'Miss'

full$Title[full$Title == 'Mme']         <- 'Mrs' 

full$Title[full$Title %in% rare_title]  <- 'Rare Title'



# Show title counts by sex again

table(full$Sex, full$Title)
full$Surname <- sapply(full$Name,  

                      function(x) strsplit(x, split = '[, ]')[[1]][1])

full$Surname                          
length(full$Surname)