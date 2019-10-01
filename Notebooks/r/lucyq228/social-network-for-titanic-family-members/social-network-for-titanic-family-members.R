library(dplyr)

library(igraph)



#read data

train <- read.csv("../input/train.csv")

test <- read.csv("../input/test.csv")



train$cat <- "train"

test$cat <- "test"



#assign avg. survival rate from training set to test set 

test$Survived <- round(sum(train$Survived)/nrow(train),2)

dat_all <- rbind(train, test)



#data cleaning and missing value imputation (very simple approach)

dat_all[is.na(dat_all$Age)==TRUE,"Age"] <- mean(dat_all$Age, na.rm=TRUE)

#assign missing Embarked to "S" (majority values)

dat_all[dat_all$Embarked=='', "Embarked"] <- "S"



head(dat_all,2)
arrange(dat_all[dat_all$Ticket %in% c("2666", "230136"),

                names(dat_all)%in%c("PassengerId", "Ticket", "Survived", "Name","cat")],Ticket)

#create with_fam variable based on if Ticket was shown more than once. 



ticket_freq <- as.data.frame(table(dat_all$Ticket))

ticket_freq$with_fam <- ifelse(ticket_freq$Freq >1, 1, 0)

colnames(ticket_freq)[1] <- "Ticket"



dat_all <- merge(dat_all, ticket_freq[, c(1,3)], by="Ticket", all.x = TRUE)



arrange(dat_all[dat_all$Ticket %in% c("2666", "110564"),

                names(dat_all)%in%c("PassengerId", "Ticket", "Name","cat", "with_fam")],Ticket)

#second var: famimly member survival rate

#first pick one ticket, then based on ticket, pick passengers that associated based on ticket

#one by one to calculate their survival rate

relative <- data.frame()

for (j in (1:nrow(ticket_freq))){

  test_member <- dat_all[dat_all$Ticket==ticket_freq[j,1] ,]

  

  for (i in (1:nrow(test_member))){

    if (nrow(test_member) == 0) {

      relative2 <- data.frame(dat_all[dat_all$Ticket==ticket_freq[j,1], "PassengerId"], 

                              ticket_freq[j,"Ticket"], NaN) #have no relative

      colnames(relative2) <- c("test_member.i...PassengerId..", "test_member.i...Ticket..", "survive")

    } 

    else {#have relative

      sub <- subset(test_member, !(test_member$PassengerId %in% test_member$PassengerId[i]))

      survive <- sum(sub[, "Survived"]) / nrow(sub)

      relative2 <- data.frame(test_member[i,"PassengerId"], test_member[i,"Ticket"],survive)

      

    }

    relative <- rbind(relative, relative2)

  }

}



colnames(relative) <- c("PassengerId", "Ticket", "FM_Survive")



#merge back with dat_all

dat_all <- merge(dat_all, relative[,c(1,3)], by="PassengerId", all.x = TRUE)



#deal with NaN for single passengers in test dataset. 

#Assign general single survival rate based on training

single_survive <-

  sum(dat_all[dat_all$with_fam==0 & dat_all$cat=="train", "Survived"])/

  length(dat_all[dat_all$with_fam==0 & dat_all$cat=="train", "Survived"])



dat_all[dat_all$with_fam==0, "FM_Survive"] <- single_survive



arrange(dat_all[dat_all$Ticket %in% c("2666", "230136"),

                names(dat_all)%in%c("PassengerId", "Ticket", "Survived", "Name","cat", "with_fam",

                                    "FM_Survive")],Ticket)
set.seed(111)



#random select 10 ticket number based on each family size (ticket frequency)

ran_ticket <- ticket_freq %>% group_by(Freq) %>% sample_n(10, replace=TRUE)

ran_ticket <- unique(ran_ticket$Ticket)



network <- dat_all[dat_all$Ticket%in%ran_ticket,

                   names(dat_all)%in%c("PassengerId", "Ticket", "Survived", "cat", "FM_Survive")]





#transform dataset to igraph format ("from", "to")

member <- data.frame()

for (g in (1: length(unique(network$Ticket)))){

  FM <- network[network$Ticket==unique(network$Ticket)[g],"PassengerId"]



  for (k in (1:length(FM))){

    member2 <- data.frame(FM[1],FM[k])

    member <- rbind(member, member2)

  }

}



colnames(member) <- c("from", "to")

member <- member[member$from!=member$to,] #remove "from" "to" same value row #data

nodes <- network[,c("PassengerId", "Survived")] #serve as data description



#transform "survived" into categorical data.Survived=3, unknown=2, deceased=1

nodes$Survived <- ifelse(nodes$Survived==1,3,

                ifelse(nodes$Survived==0,1,2)) 



#graph

net <- graph_from_data_frame(d=member, vertices = nodes, directed = T)



plot(net, edge.arrow.size=.1,

     vertex.color=c("gray50", "gold","green")[V(net)$Survived],

     vertex.size=10, vertex.label.cex=0.8)

train <- dat_all[dat_all$cat=="train",]

test <- dat_all[dat_all$cat=="test",]



library(caret)

set.seed(111)

part <- createDataPartition(train$Survived, p=0.8, list = FALSE)

ttrain <- train[part,!names(train)=="cat"]

ttest <- train[-part,!names(train)=="cat"]



ttrain$Survived <- as.factor(ttrain$Survived)

ttest$Survived <- as.factor(ttest$Survived)

model_glm <- train(Survived~., data=ttrain[,names(ttrain)%in%c("Survived",

                                                              "Pclass",

                                                              "Age",

                                                              "Fare",

                                                              "with_fam",

                                                              "FM_Survive")], 

                   method="glm", family="binomial")



summary(model_glm)

varImp(model_glm)