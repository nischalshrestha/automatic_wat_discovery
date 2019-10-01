library('ggplot2')

library('ggthemes')

library('scales')

library('dplyr')

library('mice')

library('randomForest')

library(h2o)

train <- read.csv('../input/train.csv',stringsAsFactors=F)

test<-read.csv('../input/test.csv',stringsAsFactors=F)

full<- bind_rows(train,test)

full

#str(full)
full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)

full$Title 

#grep('Capt',full$Title)
