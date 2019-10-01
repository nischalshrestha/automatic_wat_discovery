# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library('ggplot2') # visualization

library('ggthemes') # visualization

library('scales') # visualization

library('dplyr') # data manipulation

library('mice') # imputation

library('randomForest') # classification algorithm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



list.files("../input")



# Any results you write to the current directory are saved as output.


train <- read.csv('../input/train.csv', stringsAsFactors = F)

test  <- read.csv('../input/test.csv', stringsAsFactors = F)
str(train)
full  <- bind_rows(train, test) # bind training & test data

full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)

table(full$Sex, full$Title)

rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 

                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')

full$Title[full$Title == 'Mlle']        <- 'Miss' 

full$Title[full$Title == 'Ms']          <- 'Miss'

full$Title[full$Title == 'Mme']         <- 'Mrs' 

full$Title[full$Title %in% rare_title]  <- 'Rare Title'

table(full$Sex, full$Title)

full$Fsize <- full$SibSp + full$Parch + 1

summary(full$Fsize)
summary(full$Fsize)