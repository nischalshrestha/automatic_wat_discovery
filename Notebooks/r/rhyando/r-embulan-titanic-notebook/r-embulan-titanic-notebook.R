### from Arda Yildrim Solution ###



###############################

######  READ ALL SOURCE  ######

###############################



list.of.packages <- c("stringr","data.table","taRifx","methods","rgdal","sp","maptools","zoo","reshape2","plyr","rgeos","reshape","tidyr","ggplot2","rpart","rpart.plot")

new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]

if(length(new.packages)) install.packages(new.packages,repos="http://cran.r-project.org")

lapply(list.of.packages, library, character.only=T)

    

### Read data ###

test<-read.csv("../input/test.csv",sep=",",header=TRUE)

train<-read.csv("../input/train.csv",sep=",",header=TRUE)

    

head(train)

# there are many blank data in the dataset, firstly, we need to generalized the missing value #



### Tidy up NA data ###

test$Survived<-NA

combine<-rbind(test,train)

combine.r<-combine

combine<-as.data.frame(apply(combine, 2, function(x) gsub("^$|^ $", NA, x)))

    

### See NA value ###

sapply(combine,function(x) sum(is.na(x)))

par(mar = c(7,4,4,2) + 0.1)

barplot(sapply(combine,function(x) sum(is.na(x))),las=3,mar=c(5,4,4,2),main="NA Value Count")
# if you see on the plot or summary produced on the code above, there are 263,1,1014,2,and 418 missing value on Age, Fare, Cabin, Embarked, and Survived #

# Except survived parameter, whom we will find the result on, we will make an assumption based on causal relationship from the other parameter #

# first, to fill in the fare column, we take Pclass as our reference because it showed causal relationship with fare #



### fill in NA in Fare ###

combine$Pclass<-as.character(combine$Pclass)

combine$Fare<-as.numeric(as.character(combine$Fare))

freq<-combine[,c("Pclass","Fare")]

ggplot(freq, aes(as.factor(Pclass), Fare)) + geom_boxplot()

# based on the plot at this step, there is a causal relationship for Pclass and Fare, so we use median value of Fare based on Pclass to fill in missing value in fare #

combine$fare.fill<-combine$Fare

combine[is.na(combine$fare.fill),]$fare.fill<-median(combine[combine$Pclass==combine[is.na(combine$fare.fill),]$Pclass,]$fare.fill,na.rm=TRUE)
### separate the tittle ###

combine$tittle<-unlist(strsplit(as.character(combine$Name), ","))[c(F,T)]

combine$tittle<-substr(combine$tittle,2,str_locate(combine$tittle,"[.]")[,1])



### tittle and age boxplot ###

combine$tittle<-as.character(combine$tittle)

combine$Age<-as.numeric(as.character(combine$Age))

freq<-combine[,c("tittle","Age")]

ggplot(freq, aes(as.factor(tittle), Age),las=3) + geom_boxplot()+theme(axis.text.x = element_text(angle = 90, hjust = 1))



### fill in NA in Age ###

combine$Age<-as.numeric(as.character(combine$Age))

sumif<-aggregate(Age ~ tittle, data=combine, median)

names(sumif)[names(sumif)=="Age"]<-"mean.tittle"

combine$rownum<-c(1:nrow(combine))

combine<-merge(combine,sumif,by="tittle",all.x=T,sort=FALSE)

combine<-combine[order(combine$rownum),]

combine$age.fill<-ifelse(is.na(combine$Age),round(combine$mean.tittle,0),combine$Age)
### fill in NA in embark ###

combine$Embarked<-as.character(combine$Embarked)

combine$embarked.fill<-combine$Embarked

temp<-data.frame(table(combine$embarked.fill))

combine[is.na(combine$embarked.fill),]$embarked.fill<-as.character(temp[temp$Freq==max(temp$Freq),]$Var1)
### total relative calculation ###

combine$SibS<-as.numeric(as.character(combine$SibS))

combine$Parch<-as.numeric(as.character(combine$Parch))

combine$total.relative<-combine$SibS+combine$Parch



### tittle and survived plot ###

freq<-as.data.frame(table(combine[,c("tittle","Survived")]))

ggplot(freq, aes(tittle, Freq)) + geom_bar(aes(fill = Survived),stat="identity", position = "dodge")+theme(axis.text.x = element_text(angle = 90, hjust = 1))



### sex and survived plot ###

freq<-as.data.frame(table(combine[,c("Sex","Survived")]))

ggplot(freq, aes(Sex, Freq)) + geom_bar(aes(fill = Survived),stat="identity", position = "dodge")



### age range and survived plot ###

combine$age.range<-ifelse(combine$age.fill<5,"TODDLER",ifelse(combine$age.fill<10,"HIGHSCHOOL",ifelse(combine$age.fill<21,"TEEN",ifelse(combine$age.fill<35,"ADULT",ifelse(combine$age.fill<55,"OLDADULT",ifelse(combine$age.fill<200,"ELDERLY",))))))

freq<-as.data.frame(table(combine[,c("age.range","Survived")]))

ggplot(freq, aes(age.range, Freq)) + geom_bar(aes(fill = Survived),stat="identity",, position = "dodge") + scale_x_discrete(limits=c("TODDLER","HIGHSCHOOL","TEEN","ADULT","OLDADULT","ELDERLY"))



### Pclass and survived plot ###

freq<-as.data.frame(table(combine[,c("Pclass","Survived")]))

ggplot(freq, aes(Pclass, Freq)) + geom_bar(aes(fill = Survived),stat="identity", position = "dodge")



### Fare range and survived plot ###

combine$fare.range<-ifelse(combine$fare.fill==0,"FREE",ifelse(combine$fare.fill<=as.numeric(quantile(combine[combine$fare.fill>0,]$fare.fill)[2]),"CHEAP",ifelse(combine$fare.fill<=as.numeric(quantile(combine[combine$fare.fill>0,]$fare.fill)[3]),"CHEAP-MODERATE",ifelse(combine$fare.fill<=as.numeric(quantile(combine[combine$fare.fill>0,]$fare.fill)[4]),"MODERATE-EXPENSIVE",ifelse(combine$fare.fill<=as.numeric(quantile(combine[combine$fare.fill>0,]$fare.fill)[5]),"EXPENSIVE","")))))

freq<-as.data.frame(table(combine[,c("fare.range","Survived")]))

ggplot(freq, aes(fare.range, Freq)) + geom_bar(aes(fill = Survived),stat="identity", position = "dodge")



### total relative range and survived plot ###

combine$total.relative.range<-ifelse(combine$total.relative==0,"NO_RELATIVE",ifelse(combine$total.relative<=as.numeric(quantile(combine[combine$total.relative>0,]$total.relative)[2]),"SMALL_FAMILY",ifelse(combine$total.relative<=as.numeric(quantile(combine[combine$total.relative>0,]$total.relative)[3]),"SMALL-MODERATE_FAMILY",ifelse(combine$total.relative<=as.numeric(quantile(combine[combine$total.relative>0,]$total.relative)[4]),"MODERATE-BIG_FAMILY",ifelse(combine$total.relative<=as.numeric(quantile(combine[combine$total.relative>0,]$total.relative)[5]),"BIG_FAMILY")))))

freq<-as.data.frame(table(combine[,c("total.relative.range","Survived")]))

ggplot(freq, aes(total.relative.range, Freq)) + geom_bar(aes(fill = Survived),stat="identity", position = "dodge")+theme(axis.text.x = element_text(angle = 90, hjust = 1))+ scale_x_discrete(limits=c("NO_RELATIVE","SMALL_FAMILY","SMALL-MODERATE_FAMILY","MODERATE-BIG_FAMILY","BIG_FAMILY"))



### embarked and survived plot ###

freq<-as.data.frame(table(combine[,c("embarked.fill","Survived")]))

ggplot(freq, aes(embarked.fill, Freq)) + geom_bar(aes(fill = Survived),stat="identity", position = "dodge")



### ctree ###

test.edit<-combine[c(1:nrow(test)),]

train.edit<-combine[c(nrow(test)+1:nrow(train)),]

test.edit[test.edit$tittle=="Dona.",]$tittle<-"Lady."

my_tree <- rpart(Survived ~  Sex + tittle + age.fill + Pclass + fare.fill  + total.relative + embarked.fill , data = combine, method = "class", control=rpart.control(cp=0.0001))

prp(my_tree, type = 4, extra = 100)
my_prediction <- predict(my_tree, test.edit, type = "class")

test.edit<-cbind(test.edit,my_prediction)

test.result<-as.data.frame(cbind(test.edit$PassengerId,as.character(test.edit$my_prediction)))

names(test.result)<-c("PassengerId","Survived")

write.csv(test.result, file = 'R-EMBULAN_Titanic.csv', sep=",", row.names = FALSE)

head(test.result)

#write.csv(test.result,"C:/Users/ERHYADI/Desktop/Kaggle/Titanic/R-EMBULAN_Titanic.csv", sep=",",row.names=FALSE)