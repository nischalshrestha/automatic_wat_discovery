# Load the libraries

library(ggplot2) # Data visualization

library(dplyr, warn.conflicts = FALSE)



# Load the data

train_tbl <- tbl_df(read.csv('../input/train.csv', stringsAsFactors = FALSE))

test_tbl <- tbl_df(read.csv('../input/test.csv', stringsAsFactors = FALSE))



# What's in the files?

head(train_tbl)

head(test_tbl)
survived_all <- filter(train_tbl, Survived == 1)

head(survived_all)
survived_dudes <- filter(survived_all, Sex == 'male')

head(survived_dudes)
train_tbl %>% group_by(Sex) %>% summarise(n = n())
total <- train_tbl %>% group_by(Sex) %>% summarise(Total = n())

survived <- train_tbl %>% filter(Survived == 1) %>% group_by(Sex) %>% summarise(Survived = n())

perished <- train_tbl %>% filter(Survived == 0) %>% group_by(Sex) %>% summarise(Perished  = n())
survival_tbl <- inner_join(total,survived) %>% inner_join(perished)

survival_tbl
options(digits = 3)

survival_tbl <- mutate(survival_tbl, Survival_Chance = (Survived/Total)*100)

survival_tbl
train_tbl_with_age <- filter(train_tbl, !is.na(Age))

train_tbl_wo_age <- filter(train_tbl, is.na(Age))

total_with_age <- summarise(train_tbl_with_age, Total_With_Age = n())

train_tbl_wo_age <- summarise(train_tbl_wo_age, Total_Without_Age = n())

total_with_age

train_tbl_wo_age
train_tbl_with_age <- mutate(train_tbl_with_age, Age_Bin = ifelse(Age < 3, 'Infant', ifelse(Age < 13, 'Child', ifelse(Age < 19, 'Teenager',ifelse(Age < 26,'Young_Adult', ifelse(Age < 41,'Adult',ifelse(Age < 60,'Middle_Age','Old_Age')))))))

train_tbl_with_age <- mutate(train_tbl_with_age, Age_Bin_Order = ifelse(Age < 3, 1, ifelse(Age < 13, 2, ifelse(Age < 19, 3, ifelse(Age < 26, 4, ifelse(Age < 41, 5, ifelse(Age < 60, 6, 7)))))))

head(train_tbl_with_age)
total_age <- train_tbl_with_age %>% group_by(Age_Bin,Age_Bin_Order) %>% summarise(Total = n())

survived_age <- train_tbl_with_age %>% filter(Survived == 1) %>% group_by(Age_Bin,Age_Bin_Order) %>% summarise(Survived = n())

perished_age <- train_tbl_with_age %>% filter(Survived == 0) %>% group_by(Age_Bin,Age_Bin_Order) %>% summarise(Perished  = n())



survival_age_tbl <- inner_join(total_age,survived_age) %>% inner_join(perished_age)

survival_age_tbl <- mutate(survival_age_tbl, Survival_Chance = (Survived/Total)*100)





select(arrange(survival_age_tbl, Age_Bin_Order), -Age_Bin_Order)
train_tbl_with_age <- select(train_tbl_with_age,PassengerId, Survived, Age_Bin, Sex, Age,  Age_Bin_Order)

#train_tbl_with_age <- tbl_df(train_tbl_with_age)

train_tbl_with_age_by_Sex_Age_Total <- group_by(train_tbl_with_age, Sex, Age_Bin, Age_Bin_Order)

train_tbl_with_age_by_Sex_Age_Survived <- group_by(filter(train_tbl_with_age, Survived ==1), Sex, Age_Bin, Age_Bin_Order)



by_Sex_Age_Total <- summarise(train_tbl_with_age_by_Sex_Age_Total, Total = n())

by_Sex_Age_Survived <- summarise(train_tbl_with_age_by_Sex_Age_Survived, Total = n())



by_Sex_Age <- inner_join(by_Sex_Age_Total,by_Sex_Age_Survived, by = c("Sex", "Age_Bin"))

by_Sex_Age <- mutate(by_Sex_Age, Chance = (Total.y/Total.x)*100)

select(arrange(by_Sex_Age, Age_Bin_Order.y),Age_Bin,Sex, Chance, -Age_Bin_Order.y, -Age_Bin_Order.x, -Total.x, -Total.y)
summarise(filter(test_tbl, is.na(Age)), bad_age = n())
summarise(filter(test_tbl, is.na(Age), Sex == "male" ), bad_age = n())
summarise(filter(test_tbl), Total = n())
summarise(filter(test_tbl, is.na(Age), Sex == "male", Parch == 0), bad_age = n())
submit <- tbl_df(select(test_tbl,PassengerId, Sex, Age))



submit <- mutate(submit, Survived = ifelse(Sex == "female", 1, ifelse(Age < 13, 1,0)))

submit$Survived[is.na(submit$Survived)] <- 0

submit <- select(submit,PassengerId,Survived )

head(submit)

#write.csv(submit,file = "submit.csv", row.names = F)
summarise(filter(train_tbl, is.na(Age), Survived == 1, Sex == 'male'), male_age_NA_Survived = n())

summarise(filter(train_tbl, is.na(Age), Sex == 'male'), male_age_NA_Total = n())
(16/124)*100