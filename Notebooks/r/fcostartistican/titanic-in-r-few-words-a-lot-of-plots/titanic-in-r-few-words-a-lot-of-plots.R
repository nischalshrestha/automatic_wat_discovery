#Load libraries

library(ggplot2) # Data visualization

library(readr) # CSV file I/O, e.g. the read_csv function

library(caret) # Machine learning toolbox

library(tidyverse) #Data manipulation

library(magrittr) #Make code easier to read 

library(repr) #Control plot size

library(stringr) #Manipulate character variables

library(purrr) #Apply one function to a lot of elements (like sapply function but better)



#Set plot properties and colors

theme_set(theme_bw()) # Minimal theme

cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00",

               "#CC79A7", 'magenta')

options(repr.plot.width=12, repr.plot.height = 3) #Size of the plots
#Load data

train <- read.csv('../input/train.csv')

test <- read.csv('../input/test.csv')



#Take a look at the data

head(train)
train_na <- sapply(train, function(x)sum(is.na(x)|x==""))#Count the NA's in each row 

t(train_na[train_na>0]) #Print values greater than 0

test_na <- sapply(test, function(x)sum(is.na(x)|x=="")) #test  

t(test_na[test_na>0]) #Print values greater than 0
( sum(train$Survived) / nrow(train) ) %>% round(4)  * 100 #  data %>% mean (is equivalent to) mean(data) 
#Sex, Pclass and Survived

ggplot(data = train, aes( x =  Pclass,  fill = as.factor(Survived )))+ #Define the x and fill value

    #I use fill instead of color because color refers only to the perimeter of the bars and fill to the

    # inside of each figure

        geom_bar()  + # Add the bar plot

        facet_grid(~Sex) + #Create one bar plot per each Sex category

        scale_fill_manual(values = cbPalette) # Change the default colors

ggplot(data = train, aes( x =  SibSp + Parch,  fill = as.factor(Survived) ) ) +

geom_bar(position = 'dodge') + # position dodge provides one column for each SUrvived group

scale_fill_manual(values = cbPalette)
ggplot(data = train, aes( x =  Age, fill =as.factor(Survived) ) )+

geom_density(colour = 'transparent') + #Like a histogram but using continious values in 

facet_wrap(~Sex +Survived , scales = 'free_y' )+ #One plot for each Sex factor and each Survived value

scale_fill_manual(values = cbPalette)

#Warning message appears due missing values on the Age column
ggplot(data = train, aes(x =Pclass , fill = as.factor(Survived))) +

        geom_bar() +

        facet_wrap(~Embarked, scales = 'free_y') +

        scale_fill_manual(values = cbPalette)
ggplot(data = train , aes(x = as.factor(Pclass), y = Fare, colour = Sex)) +

geom_boxplot() + #Boxplot

scale_y_log10() + #Transform the scale to see better details (Warning: Objects look closer than they are)

scale_color_manual(values = cbPalette)
ggplot(data = train , aes(x = as.factor(Pclass), y = Fare, colour = as.factor(Survived))) +

geom_point(position ='jitter', size = .5) + # plot points (jitter will create an artificial random separation between points)

facet_wrap(~Sex)+

scale_color_manual(values  = cbPalette)
train$Cabin_recorded <- 'Yes'

train$Cabin_recorded[is.na(train$Cabin) |train$Cabin==''] <- "No"



ggplot(data= train, aes(x = as.factor(Pclass), fill = Cabin_recorded)) + 

geom_bar() +

facet_wrap(~Survived)+

scale_fill_manual(values = cbPalette)


train$Cabin_letter <- str_sub(train$Cabin, 1,1) %>% as.factor



Cabin_percentage <- train %>% 

                    group_by(Cabin_letter) %>% 

                    summarize(

                            count = n(),

                            Survived_pct = sum(Survived)/count)



ggplot(data= Cabin_percentage, aes(y = Survived_pct, x = Cabin_letter)) + 

geom_point(aes(size = count), colour = cbPalette[3]) +

geom_hline(data=train, aes(yintercept= mean(Survived)), colour = cbPalette[4], linetype = 2) +

geom_text(aes(x = 'D', y=.35), label = 'Survival average', colour = cbPalette[4])
#Create a function that will give the survival rate of all the passengers youner than a certain age

group_babies <- function(age){ 

    train %>% select(Age, Survived) %>% filter(Age<=age) %>% summarise(

    count=sum(Age==age),

    Age = age,

    Survival_rate = mean(Survived))

    }

#Use the function on ages from 1 to 10

babies_survival_rate <- map_df(c(1:10), group_babies)



ggplot(data = babies_survival_rate, aes(x = Age, y = Survival_rate)) +

geom_smooth(color = cbPalette[4], se =F, method ='loess', formula = 'y~x') + # Get a nice curve that 

geom_point(aes(size = count), color =cbPalette[3]) 
#To get the title of each passenger we will extract all the words with a "." in their name



library(stringr)



train %<>% mutate(

Title = str_extract(string = Name, pattern = '[A-Z][a-z]*\\.') %>% as.character,

Title = ifelse(Title%in%c('Dr.', 'Master.', 'Miss.', 'Mr.', 'Mrs.', 'Rev.'), Title, 'Other')

                   ) 

ggplot(data= train, aes(x =Title)) +

geom_bar(aes(fill=as.factor(Survived))) +

scale_fill_manual(values= cbPalette[2:3]) + 

coord_flip()
train %<>% mutate(

            Family_size = SibSp + Parch ,

            Alone = ifelse(Family_size==0,1,0),

            Head_of_family = ifelse(Parch>1 & Title == 'Mr.', 'Head of family', 'Other' ))



Family_rate <- train %>% 

                group_by(Family_size) %>% 

                summarise(

                    count = n(),

                    Survival_rate = sum(Survived)/count

                )



ggplot(data = Family_rate, aes(x = Family_size, y = Survival_rate)) +

    geom_hline(yintercept = mean(train$Survived), colour = cbPalette[4], linetype = 2) +

    geom_text(label = 'Overall Survival Rate', x= 5, y = .45, colour = cbPalette[4])+

    geom_point(aes(size = count), colour= cbPalette[2]) 
ggplot(train %>% filter(Alone!=1), aes(x = Family_size)) +

geom_bar(aes(fill = as.factor(Survived))) +

facet_wrap(~Pclass) + 

scale_fill_manual(values = cbPalette)
ggplot(train %>% filter(Alone==0), aes(x = Head_of_family, fill= as.factor(Survived))) +

geom_bar() + scale_fill_manual(values = cbPalette)

Head_family_summary <- train %>% filter(Title =="Mr.") %>% 

        group_by(Head_of_family) %>% 

        summarise(

            count = n(),

            Survival_rate = sum(Survived)/count)



Head_family_summary

ggplot(data = Head_family_summary, aes(x = Head_of_family )) +

geom_point(aes(y =Survival_rate, size = count), stat ='identity', colour = cbPalette[2])
for_test <- train %>% filter(Title =="Mr.") #Get all the Mr. from the dataset

prop.test(table(for_test$Head_of_family , for_test$Survive)) #Proportion test from the 2 groups.