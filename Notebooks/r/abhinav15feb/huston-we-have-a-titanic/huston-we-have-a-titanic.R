library('ggplot2')

library('ggthemes')

library('scales')

library('dplyr')

library('mice')
trains <- read.csv('../input/train.csv',stringsAsFactors=FALSE)

test <- read.csv('../input/test.csv',stringsAsFactors=FALSE)

full <- bind_rows(trains,test)
full$Status<- (ifelse((full$Survived=='1'), 'Survived',ifelse((full$Survived=='0'),'Dead','Not known')))

table(full$Status,full$Sex)
#table(full$Status,full$Age)

full$AgeCategory <- ifelse(full$Age<5,'Baby',ifelse(full$Age<18,'Child',ifelse(full$Age<60,'Adult','Sr.Citizen')))

table(full$Status,full$AgeCategory)
full$Title <- gsub('(.*,)|(\\..*)' ,'', full$Name)

table(full$Survived,full$Title)
RareTitles <-c('Capt','Col','Dona','Dr','Jonkheer','Lady','Major','Sir','the Countess')

full$Title[full$Title == 'Mlle']        <- 'Miss' 

full$Title[full$Title == 'Ms']          <- 'Miss'

full$Title[full$Title == 'Mme']         <- 'Mrs' 

full$RareTitle<- full$Title

full$RareTitle[full$Title %in% RareTitles]  <- 'Rare Title'

table(full$Survived,full$RareTitle)
rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 

                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')



# Also reassign mlle, ms, and mme accordingly

full$Title[full$Title == 'Mlle']        <- 'Miss' 

full$Title[full$Title == 'Ms']          <- 'Miss'

full$Title[full$Title == 'Mme']         <- 'Mrs' 

# Show title counts by sex again

#table(full$Sex, full$RTitle

full$Title[full$Title %in% rare_title]  <- 'Rare Title'



# Show title counts by sex again

table(full$Sex, full$Title)