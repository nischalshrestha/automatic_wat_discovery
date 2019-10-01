# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages

# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats

# For example, here's several helpful packages to load in 



library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv functionion

library(data.table)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



system("ls ../input")



# Any results you write to the current directory are saved as output.
train<-data.table(read.csv("../input/train.csv"))

test<-data.table(read.csv("../input/test.csv"))
#miss<-train[grepl("Miss",Name),mean(Age,na.rm=T)]

#master<-train[grep("Master",Name),mean(Age,na.rm=T)]

#mrs<-train[grep("Mrs",Name),mean(Age,na.rm=T)]

#mr<-train[grep("Mr",Name),mean(Age,na.rm=T)]

#train<-train[grepl("Miss",Name) & is.na(Age),Age:=miss]

#train<-train[grepl("Master",Name) & is.na(Age),Age:=master]

#train<-train[grepl("Mrs",Name) & is.na(Age),Age:=mrs]

#train<-train[grepl("Mr",Name) & is.na(Age),Age:=mr]

train<-train[Sex=='male',gender:=0]

train<-train[Sex=='female',gender:=1]

modelOne<-glm(Survived~gender+Age+Pclass,data=train)


miss<-test[grepl("Miss",Name),mean(Age,na.rm=T)]

master<-test[grep("Master",Name),mean(Age,na.rm=T)]

mrs<-test[grep("Mrs",Name),mean(Age,na.rm=T)]

mr<-test[grep("Mr",Name),mean(Age,na.rm=T)]

test<-test[grepl("Miss",Name) & is.na(Age),Age:=miss]

test<-test[grepl("Ms",Name) & is.na(Age),Age:=miss]

test<-test[grepl("Master",Name) & is.na(Age),Age:=master]

test<-test[grepl("Mrs",Name) & is.na(Age),Age:=mrs]

test<-test[grepl("Mr",Name) & is.na(Age),Age:=mr]

test<-test[Sex=='male',gender:=0]

test<-test[Sex=='female',gender:=1]

predictOne<-data.table(predict(modelOne,test))

final<-cbind(test,predictOne)

final<-final[V1<0,Survived:=0]

final<-final[V1>=0,Survived:=1]

final<-final[,.(PassengerId,Survived)]
write.csv(final,"submissionOne.csv",row.names=F)