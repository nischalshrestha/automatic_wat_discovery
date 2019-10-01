#Libraries#

library("dplyr");

# Import the titanic training dataset.

train<-read.csv("../input/train.csv");

test<-read.csv("../input/test.csv");

test$Survived<-rep("NA",nrow(test))

full<-rbind(train,test)

str(train)
#Target Variable:Categorical#

train%>%group_by(Survived)%>%count->freq;

freq$frequency<-freq$n/nrow(train)*100;

colnames(freq)[2]<-"count";

freq

barplot(freq$count,names.arg=freq$Survived)
#Pclass#

train%>%group_by(Pclass)%>%count()->freq;

freq$frequency<-freq$n/nrow(train)*100;

colnames(freq)[2]<-"count";

freq
#Sex#

train%>%group_by(Sex)%>%count()->freq;

freq$frequency<-freq$n/nrow(train);

colnames(freq)[2]<-"count";

freq
#Ticket#

train%>%group_by(Ticket)%>%count()->freq;

freq$frequency<-freq$n/nrow(train);

colnames(freq)[2]<-"count";

head(freq)

barplot(freq$count,names.arg=freq$Ticket)

NTicket<-freq[,1:2];

#Cabin#

train%>%group_by(Cabin)%>%count()->freq;

freq$frequency<-freq$n/nrow(train)*100;

colnames(freq)[2]<-"count";

head(freq)
#Embarked#

train%>%group_by(Embarked)%>%count()->freq;

freq$frequency<-freq$n/nrow(train)*100;

colnames(freq)[2]<-"count";

freq
#Fare#Numeric#

summary(train$Fare)

hist(train$Fare)
#Age#Numeric#

summary(train$Age)

hist(train$Age)
#Sibsp#

train%>%group_by(SibSp)%>%count()->freq;

freq$frequency<-freq$n/nrow(train)*100;

colnames(freq)[2]<-"count";

freq
#Parch#

train%>%group_by(Parch)%>%count()->freq;

freq$frequency<-freq$n/nrow(train)*100;

colnames(freq)[2]<-"count";

freq;
#Join NTicket and train#

colnames(NTicket)[2]<-"NAccompanied"

NTicket$NAccompanied<-NTicket$NAccompanied-1

train<-merge(train,NTicket,by="Ticket",all.x="TRUE")

head(train)
ind<-which(is.na(train$Age))

Age_train<-train[-ind,]

Impute_Age<-train[ind,]
age_lm_mod<-lm(Age~Pclass+Sex+Fare+Embarked+NAccompanied,data=Age_train)

Imputed_age<-predict(age_lm_mod,newdata=Impute_Age)

train$Age[ind]<-as.numeric(Imputed_age)
ind<-which(is.na(train$Cabin))

levels(train$Cabin)<-c(levels(train$Cabin),"Missing")

train$Cabin[ind]<-"Missing"
par(mfrow=c(2,2))

dt<-table(train$Survived,train$Pclass)

barplot(dt,xlab="Class",beside=TRUE,col=c("red","blue"),legend=rownames(dt))

text(1.5,100,dt[1,1])

text(2.5,150,dt[2,1])

text(4.5,150,dt[1,2])

text(5.5,150,dt[2,2])

text(7.5,350,dt[1,3])

text(8.5,150,dt[2,3])

dt<-table(train$Survived,train$Sex)

barplot(dt,xlab="SEX",beside=TRUE,col=c("red","blue"),legend=rownames(dt))

text(1.5,100,dt[1,1])

text(2.5,300,dt[2,1])

text(4.5,300,dt[1,2])

text(5.5,130,dt[2,2])

dt<-table(train$Survived,train$Embarked)

barplot(dt,xlab="Port",beside=TRUE,col=c("red","blue"),legend=rownames(dt))
age<-cut(train$Age,breaks=c(0,1,seq(from=5,to=100,by=5)))

barplot(table(train$Survived,age),col=c("red","blue"),legend=rownames(table(train$Survived,age)))

table(train$Survived,is.na(train$Age))
par(mfrow=c(2,2))

barplot(table(train$Survived,train$NAccompanied),xlab="NAccompanied",col=c("red","blue"),legend=rownames(table(train$Survived,train$NAccompanied)))

barplot(table(train$Survived,train$SibSp),xlab="SibSp",col=c("red","blue"),legend=rownames(table(train$Survived,train$SibSp)))

barplot(table(train$Survived,train$Parch),xlab="ParCh",col=c("red","blue"),legend=rownames(table(train$Survived,train$Parch)))
train%>%select(Fare,Survived)%>%group_by(Survived)%>%summarise(mean(Fare),sd(Fare),min(Fare),max(Fare))
ind<-which(train$Embarked=="")

train[ind,]
train%>%filter(Sex=="female",Pclass=="1",NAccompanied==1,Fare>=75,Fare<=85)%>%group_by(Embarked)
ind<-which(train$Embarked=="")

train$Embarked[ind]<-"C"

full$Embarked[ind]<-"C"
train$Survived<-as.logical(train$Survived)

train$Male<-as.logical(train$Sex=="male")

train$FirstClass<-as.logical(train$Pclass==1)

train$ThirdClass<-as.logical(train$Pclass==3)

train$Embarked_C<-as.logical(train$Embarked=="C")

train$Embarked_Q<-as.logical(train$Embarked=="Q")

train$Age<-cut(as.numeric(train$Age),breaks=c(0,1,seq(from=5,to=100,by=5)))

train_clean<-train[,c("Survived","Male","FirstClass","ThirdClass","Embarked_C","Embarked_Q","Age","Cabin","Fare","NAccompanied","Parch","SibSp")]

head(train_clean)
#Split Data#

set.seed(100)

ind<-sample(1:nrow(train_clean),0.85*nrow(train_clean))

train_clean<-train_clean[ind,]

valid_clean<-train_clean[-ind,]

train_clean<-train_clean[,]
#After running the first model, removing the unsignificant variables#

train_clean<-train_clean[,-8]
library(e1071)

library(rpart)

log_model<-glm(Survived~.,data=train_clean,family=binomial(link='logit'))

svm_model<-svm(factor(Survived)~.,data=train_clean,kernel='linear')

dtree_model<-rpart(Survived~.,data=train_clean,method="class")
summary(log_model)
#Train with only significant variable#

log_model_sig<-glm(Survived~.,data=train_clean[,-c(5,6,8,9,10)],family=binomial(link='logit'))

summary(log_model_sig)
library(ROCR)

library(gplots)

Predict_lm<-predict(log_model,newdata=valid_clean,type='response')

Predict_svm<-predict(svm_model,newdata=valid_clean)

Predict_dt<-predict(dtree_model,newdata=valid_clean)

Predict_lm<-ifelse(Predict_lm>0.5,TRUE,FALSE)

Predic_svm<-as.logical(Predict_svm)

Predict_dtree<-ifelse(Predict_dt[,2]>0.5,TRUE,FALSE)

Predict_lm_sig<-predict(log_model_sig,newdata=valid_clean,type='response')

Predict_lm_sig<-ifelse(Predict_lm_sig>0.5,TRUE,FALSE)
misClasificError<-mean(Predict_dtree != valid_clean$Survived)

print(paste('Accuracy Dtree',1-misClasificError))

misClasificError<-mean(Predict_svm != valid_clean$Survived)

print(paste('Accuracy SVM',1-misClasificError))

misClasificError<-mean(Predict_lm != valid_clean$Survived)

print(paste('Accuracy Logit',1-misClasificError))

misClasificError<-mean(Predict_lm_sig!= valid_clean$Survived)

print(paste('Accuracy Logit Sig',1-misClasificError))
par(mfrow=c(2,2))

Predict_lm<-predict(log_model,newdata=valid_clean,type='response')

Predict_svm<-as.numeric(predict(svm_model,newdata=valid_clean))

Predict_dtree<-predict(dtree_model,newdata=valid_clean)[,2]

Predict_lm_sig<-predict(log_model_sig,newdata=valid_clean,type='response')

pr_dt <- prediction(Predict_dtree, valid_clean$Survived)

prf_dt <- performance(pr_dt, measure = "tpr", x.measure = "fpr")

plot(prf_dt)

pr_svm <- prediction(Predict_svm, valid_clean$Survived)

prf_svm <- performance(pr_svm, measure = "tpr", x.measure = "fpr")

plot(prf_svm)

pr_lm <- prediction(Predict_lm, valid_clean$Survived)

prf_lm <- performance(pr_lm , measure = "tpr", x.measure = "fpr")

plot(prf_lm)

pr_lm_sig <- prediction(Predict_lm_sig, valid_clean$Survived)

prf_lm_sig <- performance(pr_lm_sig , measure = "tpr", x.measure = "fpr")

plot(prf_lm_sig)





auc_dt <- performance(pr_dt, measure = "auc")

auc_dt <- auc_dt@y.values[[1]]

paste("AUC DT: ",auc_dt)

auc_svm <- performance(pr_svm, measure = "auc")

auc_svm <- auc_svm@y.values[[1]]

paste("AUC SVM: ",auc_svm)

auc_lm <- performance(pr_lm, measure = "auc")

auc_lm <- auc_lm@y.values[[1]]

paste("AUC LM: ",auc_lm)

auc_lm <- performance(pr_lm_sig, measure = "auc")

auc_lm <- auc_lm@y.values[[1]]

paste("AUC LM: ",auc_lm)
Predict_lm_sig<-predict(log_model_sig,newdata=valid_clean,type='response')

Predict_lm_sig<-ifelse(Predict_lm_sig>0.55,TRUE,FALSE)

misClasificError<-mean(Predict_lm_sig!= valid_clean$Survived)

print(paste('Accuracy Logit Sig',1-misClasificError))
#Load Test Set#

test<-read.csv("../input/test.csv");

summary(test)
#missing value#

ind<-which(is.na(full$Fare))

full[ind,]

full[-ind,]%>%filter(Pclass==3,Sex=="male",Age>=30,Embarked=="S")%>%summarise(mean(Fare))

full$Fare[ind]<-11.4508
#variable transformation for full set#

full%>%group_by(Ticket)%>%count()->freq;

freq$frequency<-freq$n/nrow(full);

colnames(freq)[2]<-"count";

NTicket<-freq[,1:2];

colnames(NTicket)[2]<-"NAccompanied"

NTicket$NAccompanied<-NTicket$NAccompanied-1

full<-merge(full,NTicket,by="Ticket",all.x="TRUE")
ind<-which(is.na(full$Age))

Age_full<-full[-ind,]

Impute_Age<-full[ind,]

age_lm_mod<-lm(Age~Pclass+Sex+Fare+Embarked+NAccompanied,data=Age_full)

Imputed_age<-predict(age_lm_mod,newdata=Impute_Age)

full$Age[ind]<-as.numeric(Imputed_age)
full$Male<-as.logical(full$Sex=="male")

full$FirstClass<-as.logical(full$Pclass==1)

full$ThirdClass<-as.logical(full$Pclass==3)

full$Embarked_C<-as.logical(full$Embarked=="C")

full$Embarked_Q<-as.logical(full$Embarked=="Q")

full$Age<-cut(as.numeric(full$Age),breaks=c(0,1,seq(from=5,to=100,by=5)))

full_clean<-full[,c("Male","FirstClass","ThirdClass","Embarked_C","Embarked_Q","Age","Fare","NAccompanied","Parch","SibSp")]

head(full_clean)
test_clean<-full_clean[-(1:891),]

str(test_clean)
valid_clean<-valid_clean[,-8]
#for dataset train it on full dataset#

log_model<-glm(Survived~.,data=rbind(train_clean,valid_clean),family=binomial(link='logit'))

svm_model<-svm(factor(Survived)~.,data=rbind(train_clean,valid_clean),kernel='linear')

dtree_model<-rpart(Survived~.,data=rbind(train_clean,valid_clean),method="class")

Predict_lm<-predict(log_model,newdata=test_clean,type='response')

Predict_svm<-predict(svm_model,newdata=test_clean)

Predict_dt<-predict(dtree_model,newdata=test_clean)

output<-cbind(test$PassengerId,Predict_lm,Predic_svm,Predict_dt[,2])

path <- "../data/";

kaggle <- F;

if(!dir.exists(path)) {

        kaggle <- T;

	path <- "../input/"; # changing path to Kaggle's environment

}



if(kaggle) {

	write.csv(output, "submission.csv", row.names =F, quote=F);

} else {

	write.csv(output, "../data/submission.csv", row.names =F, quote=F);

}