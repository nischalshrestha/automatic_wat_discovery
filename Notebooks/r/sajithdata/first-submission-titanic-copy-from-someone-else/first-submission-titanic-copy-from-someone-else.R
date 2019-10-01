# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



# Any results you write to the current directory are saved as output.



library('ggplot2') # visualization

library('ggthemes') # visualization

library('scales') # visualization

library('dplyr') # data manipulation

library('mice') # imputation

library('randomForest') # classification algorithm



# The train and test data is stored in the ../input directory

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