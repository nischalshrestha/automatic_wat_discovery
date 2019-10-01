# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library('ggplot2') # Data visualization

library('ggthemes')

library('scales')



library('dplyr') #data manipulation

library('mice') #imputation

library('randomForest') #classification algo

library('readr') # CSV file I/O, e.g. the read_csv function



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



# Any results you write to the current directory are saved as output.
train <- read.csv('../input/train.csv', stringsAsFactors = F)

test <- read.csv('../input/test.csv', stringsAsFactors = F)



full <- bind_rows(train, test) #bind training and test data



str(full)
full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)



table(full$Sex, full$Title)
rare_title <- c('Capt', 'Col', 'Don', 'Dona', 'Dr', 'the Countess', 'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir')

full$Title[full$Title == 'Mlle'] <- 'Miss'

full$Title[full$Title == 'Ms'] <- 'Miss'

full$Title[full$Title == 'Mme'] <- 'Mrs'

full$Title[full$Title %in% rare_title] <- 'Rare Title'

table(full$Sex, full$Title)
full$Surname <- sapply(full$Name, 

                      function(x) strsplit(x, split = '[,.]')[[1]][1]);

cat(paste('We have <b>', nlevels(factor(full$Surname)), '</b> unique surnames. I would be interested to infer ethnicity based on surname --- another time.'))

                          

                       

                         
#family size variable: # of siblings/spouses + # of parents/children + themselves

full$Fsize <- full$SibSp + full$Parch + 1



#family variable

full$Family <- paste(full$Surname, full$Fsize, sep='_')
ggplot(full[1:891,], aes(x = Fsize, fill = factor(Survived))) +

  geom_bar(stat='Count', position='dodge') +

  scale_x_continuous(breaks=c(1:11)) +

  labs(x = 'Fam Size') +

  theme_few()
full$FsizeD[full$Fsize == 1] <- 'singleton'

full$FsizeD[full$Fsize < 5 & full$Fsize > 1] <- 'small'

full$FsizeD[full$Fsize > 4] <- 'large'



mosaicplot(table(full$FsizeD, full$Survived), main='Family Size by Survival', shade=TRUE)