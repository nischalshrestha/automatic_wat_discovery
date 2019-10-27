# TODO Normalize and add to rsnips
# subsetting on a particular column with a filter
full_data$Title[full_data$Title == 'Mlle'] 
# subsetting on main dataframe then referencing a column
full[is.na(full$embarked), ]$embarked <- 'S'
full[is.na(full$Age) & full$Title == "Mr", ]$Age
# subsetting using which with chained conditions where rhs is either numeric or string
train[which(train$Parch > 0 & train$Age > 18 & train$Name == "Mrs."), c("Mother")] <- 1
full$Mother[full$Sex == 'female' & full$Parch > 0 & full$Age > 18 & full$Title != 'Miss'] <- 'Mother'
# subsetting using chained conditions using mix of boolean operators
fulldata$isKids[fulldata$Age <=18  & (fulldata$Parch > 0 | fulldata$TicketGroupSize>1 | fulldata$FamilynameSize>1)  & fulldata$Title != 'Mrs'] <- 1


# subsetting with a mix of a function and regular condition (which, is.na, grepl, %in%)
train$farediv[which(train[, 10] > 20)] <- 2 
test$Age[is.na(test$Age) & test$Name == "Master."] <- masterage
train$Age[grepl("Master\\.", train$Name) & is.na(train$Age)] = mean.master
combi$Title[combi$Title %in% c('Mlle','Mme')]<-'Mlle'
combined.df$TitleGroup[which(combined.df$Title %in% c("Rev"))] <- "MrRev" 

# subsetting while retaining the original dimension (keeping it a data frame)
data[[i]] <- full[train_dev_idx, var[[i]], drop=F]

# function call on subsetting operation(s)
head(whole.data)
head(full_data$Name)
head(train_data[,coloumns.to.fit],10)
head(data[c("FamilyName", "Sector")])
head(subset(df,fpr<0.2))


# chained function calls
age[age_na_rows] = sample(na.omit(full$Age), length(age_na_rows))
head(subset(cutoffs, fpr < 0.2))


# inverse boolean vector using ! (note: the ! has lower precedence than the other boolean operators)
# this is in contrast to Python where the equivalent ~ is higher precedence. 
# example in R:
# > df
#   a b
# 1 1 4
# 2 2 5
# 3 3 6
# > !df$a > 1
# [1]  TRUE FALSE FALSE
# > !(df$a > 1)
# [1]  TRUE FALSE FALSE
train$Cabin[which(!train$Cabin == "")]<-1

# 
trn_m$Survived_F = factor(trn_m$Survived)
# renaming for a column where there are no missing values
full[which(is.na(full$Survived)), 2] <- 0

# Stats stuff
# double $
summary(train_imp$imp$Age)
summary(df[, c('Pclass', 'Survived')])

# na.rm is F by default whereas pandas is T by default
mean(my.test.data[,i], na.rm = TRUE)
full$Fare[1044] <- median(full[full$Pclass == 3 & full$Embarked == 'S', ]$Fare,na.rm = TRUE)

# More complex stuff that transforms columns
# creation/transformation of column based on 1 other column
mutate(rate=n/FamilySize)
# creation/transformation of column based on 2 other columns
mutate(FamSize = SibSp + Parch)