library(dplyr)

library(cluster)

library(ggplot2)



train <- read.csv('../input/train.csv', stringsAsFactors = F)

test  <- read.csv('../input/test.csv', stringsAsFactors = F)



full <- bind_rows(train,test)



# Create a family size variable including the passenger themselves

full$Fsize <- full$SibSp + full$Parch + 1



# grab surname from passenger name

full$Surname <- sapply(full$Name,  

                       function(x) strsplit(x, split = '[,.]')[[1]][1])
## Create a family variable and group passengers into family

full$family <- NaN



## filter out those family names appearing more than once

potentialNames <- full %>% 

    group_by(Surname) %>% 

    summarize(family_n = n()) %>% 

    filter(family_n > 1) %>% 

    select(Surname)
## if passengers are from the same family one would assume they share some attributes

compareAttribs <- c("Pclass", "Fsize", "Ticket", "Fare", "Embarked")



## set the minimum number of attributes that have to match in order to be qualified as a family

minMatch <- 3
# initalize a family number, which is going to be increased with every family identified

familyNum <- 1 



# loop through all potential families

for (i in 1:length(potentialNames$Surname)){

  

  # select all passengers with a given surname

  tmp <- full %>% 

    filter(Surname %in% potentialNames$Surname[i] & Fsize>1) %>% 

    select(one_of(c("PassengerId",compareAttribs))) %>% 

    mutate_each(funs(as.factor))

  

  if (nrow(tmp)>1){

    # calculate the dissimilarity

    dissim <- cluster::daisy(select(tmp,one_of(compareAttribs)))

    # group passengers according to their similarity (i.e. cluster analysis)

    hclust <- as.hclust(cluster::agnes(dissim,diss = TRUE))

    # get the clusters according to our threshold, such that passenger in clusters match at least in minMatch attributes

    clusts <- cutree(hclust, h = 1 - minMatch/length(compareAttribs))

    

    # get clusters with at least 2 members and assign a family number to the members

    finalClusts <- unique(clusts[duplicated(clusts)])

    for (j in 1:length(finalClusts)) {

      full$family[full$PassengerId %in% tmp$PassengerId[clusts==finalClusts[j]]] <- familyNum

      familyNum <- familyNum + 1

    }

  }

}
full %>% arrange(family) %>% select(PassengerId, family, Surname, Pclass, Fsize, Ticket, Fare, Embarked) %>% head(n=10)
full$quote <- "alone"

for (i in 1:familyNum){

  tmp <- full %>% filter(family==i)

  for (j in 1:nrow(tmp)) {

    quote <- mean(tmp$Survived[setdiff(1:nrow(tmp),j)], na.rm = T)

    full$quote[full$PassengerId == tmp$PassengerId[j]] <- quote

  }

}
full %>% group_by(quote) %>% summarize(msurvived = mean(Survived, na.rm=T), n=n())
full$quote[full$quote < 0.5] <- 0

full$quote[full$quote >= 0.5 & full$quote <= 1] <- 1

full$quote[is.na(full$quote)] <- NaN



dfplot <- full %>% 

    group_by(quote) %>% 

    summarize(m = mean(Survived,na.rm=T), n = n(), se = sd(Survived,na.rm=T)/sqrt(n()))



# convert quote to factor and plot

dfplot$quote <- factor(dfplot$quote, levels = c("0","1","NaN","alone"))

ggplot(dfplot,aes(x=quote,y=m,fill=quote))+geom_bar(stat="identity")+geom_errorbar(aes(x=quote,ymin=m-se,ymax=m+se,width=0.35))+

    xlab("Proportion of other family members surviving")+ylab("Probability of own survival")+

    theme_bw()+theme(axis.text = element_text(size=18), axis.title = element_text(size=22), legend.position = "none")