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
train<-data.table(read.csv("../input/train.csv",stringsAsFactors=F))

test<-data.table(read.csv("../input/test.csv",stringsAsFactors=F))
test<-test[,Survived:='none']

setcolorder(test,c(1,12,2:11))
train<-train[,Gender:=ifelse(Sex=='female',1,2)]

median_ages<-matrix(rep(0,6),nrow=2,ncol=3)

for(i in c(1,2)){

   for(j in c(1,2,3))

       median_ages[i,j]<-train[Gender==i & Pclass==j,median(Age,na.rm=T)] 

}

for(i in c(1,2)){

   for(j in c(1,2,3))

       train[Gender==i & Pclass==j & is.na(Age),Age:=median_ages[i,j]] 

}

    

    

test<-test[,Gender:=ifelse(Sex=='female',1,2)]

median_ages<-matrix(rep(0,6),nrow=2,ncol=3)

for(i in c(1,2)){

   for(j in c(1,2,3))

       median_ages[i,j]<-test[Gender==i & Pclass==j,median(Age,na.rm=T)] 

}

for(i in c(1,2)){

   for(j in c(1,2,3))

       test[Gender==i & Pclass==j & is.na(Age),Age:=median_ages[i,j]] 

}
data<-rbind(train,test)

data<-data[,Title:=unlist(strsplit(trimws(unlist(strsplit(Name,',', fixed = T))[2])," ",fixed = T))[1],by=PassengerId]

data<-data[Title=='the',Title:="Countess."]

data<-data[,HighClass:=ifelse(Title %in% c("Mr.","Mrs.","Master.","Mme.","Mlle.","Ms.","Miss."),0,1)]

train_data<-data[Survived!="none"]

train_data<-train_data[,Survived:=as.numeric(Survived)]

test_data<-data[Survived=='none']

test_data<-test_data[,Survived:=NULL]
modelTwo<-glm(Survived~Age+Pclass+Gender+HighClass,train_data,family=binomial)

predTwo<-data.table(Survived=predict(object = modelTwo,test_data))

final<-cbind(test_data$PassengerId,predTwo)

setnames(final,"V1","PassengerId")

final<-final[,Survived:=ifelse(Survived<=0,0,1)]
write.csv(final,"submissionThree.csv",row.names=F)