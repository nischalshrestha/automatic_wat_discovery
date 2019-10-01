library("ggplot2") # Data visualization

library("readr") # CSV file I/O, e.g. the read_csv function

library("ggthemes") # Data visualization

library("scales") # Data visualization

library("dplyr") # Data manipulation

library("mice") # Imputation

library("randomForest") # Classification Algorithm



system("ls ../input")



# Start with train data and read it

train <- read.csv("../input/train.csv", stringsAsFactors = F)

test <- read.csv("../input/test.csv", stringsAsFactors = F)



#combine test and train datasets

full <- bind_rows(train,test)



# check data

str(train)

str(test)

str(full)
head(full)
#Extract the Title from Name and make new variable from it

full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)



#See the title counts by sex 

table(full$Sex, full$Title)
# Combined very low cell title counts

rare_title <- c("Dona", "Lady", "the Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer")



# Reassing Mlle, Ms, Mme accordingly

full$Title[full$Title == "Mlle"]  <- "Miss"

full$Title[full$Title == "Ms"]  <- "Miss"

full$Title[full$Title == "Mme"]  <- "Mrs"

full$Title[full$Title %in% rare_title]  <- "Rare"



#See the title by sex again

table(full$Sex, full$Title)
#Extract the Surname from Name and make new variable from it

full$Surname <- sapply(full$Name,

                       function(x) strsplit(x, split = "[,.]")[[1]][1])

#Lets see our data again

head(full)
#Create a family size variable including the passenger themselves

full$Fsize <- full$SibSp + full$Parch +1



#Create a family variable

full$Family <- paste(full$Surname, full$Fsize, sep="_")
# Use ggplot to Visualize

ggplot(full[1:891,], aes(x = Fsize, fill = factor(Survived))) +

    geom_bar(stat = "count", position = "dodge") +

        scale_x_continuous(breaks=c(1:11)) +

            labs(x = "Family Size") + 

                theme_few()
#Collapse and Discretized Family Size

full$FsizeD[full$Fsize == 1] <- "Single"

full$FsizeD[full$Fsize < 5 & full$Fsize > 1] <- "Small"

full$FsizeD[full$Fsize >4 ] <- "Large"



#Show Discretized Family Size by Survival using mosaic plot

mosaicplot(table(full$FsizeD, full$Survived), main = "Family Size by Survival", shade = TRUE)
#Lets check passenger cabin (seems have a lot of missing value)

full$Cabin[1:28]
#The first character is the deck, lets take a look

strsplit(full$Cabin[2], NULL)[[1]]
#Create Deck variable

full$Deck <- factor(sapply(full$Cabin, function(x) strsplit(x, NULL)

                            [[1]][1]))
#Lets take a look at our data again

head(full)
#Passenger 62 and 830 missing embarkment

full [c(62,830),]
#Get rid of missing passenger ID

embark_fare <- full %>% 

    filter(PassengerId != 62 & PassengerId != 830)



#Use ggplot2 to visulize embarkment, passenger class and median fare

ggplot(embark_fare, aes(x = Embarked, y = Fare, fill= factor(Pclass)))+

   geom_boxplot() +

        geom_hline(aes(yintercept = 80), colour = "red", linetype = "dashed", lwd = 2) +

            scale_y_continuous(labels = dollar_format()) +

                theme_few()



#Since their fare is 80 for 1st class, they most likely to embarked from 'C', lets fill it!

full$Embarked[c(62,830)] <- "C"
#Passenger 1044 missing Fare

full[1044, ]
#Visualize fare among 3rd Class and S Embarkment

ggplot(full[full$Pclass == '3' & full$Embarked == 'S', ],

        aes(x = Fare)) +

        geom_density(fill = "#99d6ff", alpha = 0.4) +

        geom_vline (aes(xintercept = median(Fare, na.rm=T)),

                   colour = "red", linetyp = "dashed", lwd =1) +

        scale_x_continuous(labels = dollar_format()) + 

                     theme_few()
#Replacing missing value of Fare with median Fare of class/embarkment

full$Fare[1044] <- median(full[full$Pclass == "3" & full$Embarked == "S",]$Fare, na.rm = TRUE)



#Lets see our data now

str(full)
#See number of missing value on Age

sum(is.na(full$Age))
# Make variables into factors

factor_vars <- c("PassengerId", "Pclass", "Sex", "Embarked", "Title", "Surname", "Family","FsizeD")



full[factor_vars] <- lapply(full[factor_vars], function(x) as.factor(x))



# Lets see our data again

str(full)
#Set a random seed

set.seed(129)



#Perform mice imputation, excluidng less than seful variables

mice_mod <- mice(full[, !names(full) %in% c("PassengerId","Name","Ticket","Cabin","Family","Surname","Survived")], method = "rf")
#Save the complet output

mice_output <- complete(mice_mod)
#Visualize age distribution

par(mfrow = c(1,2))

hist(full$Age, freq = F, main="Age: Original Data",

    col = "darkgreen", ylim=c(0,0.04))

hist(mice_output$Age, freq=F, main="Age:Mice Output",

    col = "lightgreen", ylim=c(0,0.04))
#Replace age variable from mice model

full$Age <- mice_output$Age



# Show new number of missing Age values

sum(is.na(full$Age))
# Create the column child, and indicate whether child or adult

full$Child[full$Age < 18] <- 'Child'

full$Child[full$Age >= 18] <- 'Adult'



# Show counts of child by survival

table(full$Child, full$Survived)
# Adding Mother variable

full$Mother <- 'Not Mother'

full$Mother[full$Sex == 'female' & full$Parch > 0 & full$Age > 18 & full$Title != 'Miss'] <- 'Mother'



# Show counts of mother by survival

table(full$Mother, full$Survived)
# Finish the new variable by factorizing it

full$Child  <- factor(full$Child)

full$Mother <- factor(full$Mother)
head(full)

str(full)
md.pattern(full)
# Split the data back into a train set and a test set

train <- full[1:891,]

test <- full[892:1309,]
# Set a random seed

set.seed(754)



# Build the model (note: not all possible variables are used)

rf_model <- randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked+

                         Title + FsizeD + Child + Mother,

                         data = train)



# Show model error

plot(rf_model, ylim=c(0,0.36))

legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)
# Get importance

importance    <- importance(rf_model)

varImportance <- data.frame(Variables = row.names(importance), 

                            Importance = round(importance[ ,'MeanDecreaseGini'],2))



# Create a rank variable based on importance

rankImportance <- varImportance %>%

  mutate(Rank = paste0('#',dense_rank(desc(Importance))))



# Use ggplot2 to visualize the relative importance of variables

ggplot(rankImportance, aes(x = reorder(Variables, Importance), 

    y = Importance, fill = Importance)) +

  geom_bar(stat='identity') + 

  geom_text(aes(x = Variables, y = 0.5, label = Rank),

    hjust=0, vjust=0.55, size = 4, colour = 'red') +

  labs(x = 'Variables') +

  coord_flip() + 

  theme_few()
# Predict using the test set

prediction <- predict(rf_model, test)



# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)

solution <- data.frame(PassengerID = test$PassengerId, Survived = prediction)



# Write the solution to file

write.csv(solution, file = 'Titanic_Solution.csv', row.names = F)



# Lets take a look of solution

head(solution)