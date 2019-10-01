print(list.files("../input"))
input_df <- read.csv("../input/train.csv")

train_df <- input_df[1:500, ]

val_df <- input_df[501:891, ]



test_df <- read.csv("../input/test.csv")
nrow(train_df)

head(train_df)
library(mice)
my_mids <- mice(train_df)
my_glm <- glm(Survived ~ Age + Pclass + Sex + SibSp + Parch + Fare,

              data = train_df, family = "binomial")
summary(my_glm)
glm_predictions <- predict(my_glm, newdata = val_df, type = "response")

val_df$pred_survived <- as.numeric(glm_predictions > .5)

val_df$pred_survived <- ifelse(is.na(val_df$pred_survived), 0, val_df$pred_survived)



#print(glm_predictions)

print(val_df$pred_survived)

val_df[val_df$PassengerId == 503, ]







got_right <- with(val_df, pred_survived == Survived)

got_wrong <- with(val_df, pred_survived != Survived)



n_right <- sum(got_right, na.rm = TRUE)

n_wrong <- sum(got_wrong, na.rm = TRUE)

accuracy <- n_right / (n_right + n_wrong)

cat("accuracy: ", accuracy, "\n")





submission_df <- test_df[, c("PassengerId", "Survived")]

write.csv(submission_df, "submission.csv", row.names = FALSE)

print(list.files())
print(list.files("../input"))
input_df <- read.csv("../input/train.csv")[, c("Survived", "Pclass", "Sex", "Age", "SibSp", "Parch")]

train_df <- input_df[1:500, ]

val_df <- input_df[501:891, ]



test_df <- read.csv("../input/test.csv")
nrow(train_df)

head(train_df)

summary(train_df)
library(mice)
my_mids <- mice(train_df, m = 1, maxiter = 25)
my_glm <- glm(Survived ~ Age + Pclass + Sex + SibSp + Parch + Fare,

              data = train_df, family = "binomial")
summary(my_glm)
glm_predictions <- predict(my_glm, newdata = val_df, type = "response")

val_df$pred_survived <- as.numeric(glm_predictions > .5)

val_df$pred_survived <- ifelse(is.na(val_df$pred_survived), 0, val_df$pred_survived)



#print(glm_predictions)

print(val_df$pred_survived)

val_df[val_df$PassengerId == 503, ]







got_right <- with(val_df, pred_survived == Survived)

got_wrong <- with(val_df, pred_survived != Survived)



n_right <- sum(got_right, na.rm = TRUE)

n_wrong <- sum(got_wrong, na.rm = TRUE)

accuracy <- n_right / (n_right + n_wrong)

cat("accuracy: ", accuracy, "\n")





submission_df <- test_df[, c("PassengerId", "Survived")]

write.csv(submission_df, "submission.csv", row.names = FALSE)

print(list.files())