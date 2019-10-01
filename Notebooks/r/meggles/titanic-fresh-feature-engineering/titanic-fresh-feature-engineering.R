#load packages

library("ggplot2") # data visualization

library("caret") # multiple model training/testing functions

library("readr") # CSV file I/O, e.g. the read_csv function

library("dplyr") # several Hadley Wickham data-wrangling packages

library("mice") # imputing missing values

library("VIM") # visualizing missing values

library("stringr") # feature engineering

library("arules") # feature engineering

library("corrplot") # correlogram 



options(warn=-1) # turn warnings off 
# read in the data

train_full <- read_csv('../input/train.csv')



# train split into train and validate

inTrain <- createDataPartition(train_full$Survived, times = 1, p = 0.8, list=F)



train <- train_full[inTrain,]

val <- train_full[-inTrain,]
nrow(train) # number of training observations

nrow(val) # number of training observations
train <- mutate(train,

                Cabin_Deck = str_sub(Cabin,1,1),

                Ticket_Digit = nchar(Ticket),

                Ticket_Alpha = str_detect(Ticket, '[[:alpha:]]'),

                Family_Size = Parch+SibSp,

                Name_Family = gsub(",.*$", "", Name),

                Title = str_sub(Name, 

                                str_locate(Name, ",")[ , 1] + 2, 

                                str_locate(Name, "\\.")[ , 1] - 1)

               )

# credit to https://www.kaggle.com/c/titanic/discussion/30733 for Title regex
# subset training data to include only variables we would consider using in our model

train_sub <- select(train,

                    Survived,Pclass,Sex,Age,SibSp,Parch,Fare,Embarked,

                    Cabin_Deck,Ticket_Digit,Ticket_Alpha,Name_Family,Title,Family_Size)

# missing value pattern matrix

md.pattern(train_sub)



# visualization of missing values

train_mice_plot <- aggr(train_sub, col=c('palegreen2','palegoldenrod'),

                    numbers=T, sortVars=T,

                    labels=names(train_sub), cex.axis=.7,

                    gap=3, ylab=c("Missing data","Pattern"))
# Hypothesis: Age missing values are correlated with at least one other variable that has no missing values (aka not totally missing at random)

print('Proportion Table: TRUE = Age is missing  X  Survived = 1 if survived')

round(prop.table(table(is.na(train$Age), train$Survived), 1),2)



print('Proportion Table: TRUE = Age is missing  X  Pclass = 1 is first class')

round(prop.table(table(is.na(train$Age), train$Pclass), 1),2)



print('Distribution of Fare  X  Age missing vs present')

print('Age missing')

summary(train %>% 

        filter(is.na(Age)) %>%

        select(Fare)

        )

print('Age present')

summary(train %>% 

        filter(!is.na(Age)) %>%

        select(Fare)

        )
# Hypothesis: Cabin missing values are correlated with at least one other variable that has no missing values (aka not totally missing at random)



print('Proportion Table: TRUE = Cabin/Deck is missing  X  Survived = 1 if survived')

round(prop.table(table(is.na(train$Cabin_Deck), train$Survived), 1),2)



print('Proportion Table: TRUE = Cabin/Deck is missing  X  Pclass = 1 is first class')

round(prop.table(table(is.na(train$Cabin_Deck), train$Pclass), 1),2)



print('Distribution of Fare  X  Cabin/Deck missing vs present')

print('Cabin/Deck missing')

summary(train %>% 

        filter(is.na(Cabin_Deck)) %>%

        select(Fare)

        )

print('Cabin/Deck present')

summary(train %>% 

        filter(!is.na(Cabin_Deck)) %>%

        select(Fare)

        )
train_mm <- model.matrix(~Pclass+Sex+Age+

                         SibSp+Parch+Fare+

                         Embarked+Cabin_Deck+

                         Ticket_Digit+Ticket_Alpha+Title+Name_Family,

                         train_sub)



train_imp <- mice(train_sub, 

                  m = 1,

                  method = "cart", 

                  seed = 5, 

                  printFlag=FALSE)



# check out the distribution of imputed Age values

# imputed values look like they match the general distribution of the complete values

summary(train_imp$imp$Age) 



# merge imputed values with complete observations

train <- complete(train_imp) 



Age_hist_imp <- ggplot(train, aes(Age))

Age_hist_imp + geom_histogram(binwidth = 5) 



# check to see if any NAs in Age after imputation: none

which(is.na(train$Age)==T) 
train <- mutate(train, 

                Cabin_Deck_i = ifelse(!is.na(Cabin_Deck),

                                Cabin_Deck,

                                ifelse(Pclass == 1,

                                       'ABCD', 

                                        # not including T because only one passenger

                                        # in the training set was assigned cabin T

                                       ifelse(Pclass == 2,

                                              'E',

                                             'F'))))
# histogram of Fare by Pclass (no missing values)

Fare_hist <- ggplot(train, aes(Fare))

Fare_hist + geom_histogram(binwidth=25) + facet_grid(Pclass~.)



# subset passengers to consider first class only

train_Pclass1 <- filter(train, Pclass == 1) 



# divide the Fare distribution for Pclass 1 into 

# the number of Decks (4) that we need to break apart

# for Cabin_Deck imputation

cuts <- discretize(train_Pclass1$Fare,

                   method = 'cluster',

                   categories = 4,

                   ordered = T,

                   onlycuts = T)



train <- mutate(train, Cabin_Deck_i2 = ifelse(Cabin_Deck_i != "ABCD",

                                       Cabin_Deck_i,

                                       ifelse(Fare < cuts[2],

                                             "D",

                                             ifelse(Fare < cuts[3],

                                                   "C",

                                                   ifelse(Fare < cuts[4],

                                                         "B", 

                                                         "A")))))
train <- mutate(train, Cabin_Deck_i3 = ifelse(Cabin_Deck_i2 == 'A',1,

                                ifelse(Cabin_Deck_i2 == 'B',2,

                                      ifelse(Cabin_Deck_i2 == 'C',3,

                                            ifelse(Cabin_Deck_i2 == 'D',4,

                                                  ifelse(Cabin_Deck_i2 == 'E',5,

                                                        ifelse(Cabin_Deck_i2 == 'F',6,

                                                              ifelse(Cabin_Deck_i2 == 'G',7,8))))))))
train <- mutate(train, 

                Embarked = ifelse(is.na(Embarked),

                                 'S', Embarked))
train_cor <- select(train,

                    Age, Fare, SibSp, Parch,

                    Pclass, Cabin_Deck_i3,

                    Ticket_Digit,

                    Family_Size)

cor <- cor(train_cor)

corrplot(cor, method="number")
Deck_plot <- ggplot(train, aes(Cabin_Deck_i2))

Deck_plot + geom_bar(aes(fill = as.factor(Survived)))



round(prop.table(table(train$Cabin_Deck_i2, train$Survived), 1),2)
Family_plot <- ggplot(train, aes(Family_Size))

Family_plot + geom_histogram(aes(fill = as.factor(Survived)))
Title_plot <- ggplot(train, aes(Title))

Title_plot + geom_bar(aes(fill = as.factor(Survived)))



round(prop.table(table(train$Title, train$Survived), 1),2)