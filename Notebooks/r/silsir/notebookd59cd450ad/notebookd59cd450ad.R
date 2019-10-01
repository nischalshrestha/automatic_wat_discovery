# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function

library(ggthemes)

library(dplyr)

library(mice)

library(scales)

library(randomForest)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



# Any results you write to the current directory are saved as output.
train <- read.csv('../input/train.csv', stringsAsFactors = F)

test <- read.csv('../input/test.csv', stringsAsFactors = F)



str(train)

str(test)

full <- bind_rows(train,test) #bind_rows doesn't need same amount of variables, auto-adds and fills with NA



str(full)
full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)



#creates new variable title, by using gsub, which takes Name by identifying patterns

#after it finds the surname, and what's after the period of the title, it replaces with what's left

#in this case, that is the title



table(full$Sex, full$Title)
rare <- c('Dona', 'Lady', 'the Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer') 



full$Title[full$Title == 'Mlle'] <- 'Miss'

full$Title[full$Title == 'Ms'] <- 'Miss'

full$Title[full$Title == 'Mme'] <- 'Mrs'

full$Title[full$Title %in% rare] <- 'Rare Title'



table(full$Sex, full$Title)
full$Fam.size <- full$SibSp + full$Parch + 1



full$Family <- paste(full$Surname, full$Fam.size, sep='.....')



ggplot(full[1:891,], aes(x = Fam.size, fill = factor(Survived))) +

      geom_bar(stat = 'count', position = 'dodge') +

      scale_x_continuous(breaks=c(1:11)) +

      labs(x = 'Family Size') +

      theme_few()