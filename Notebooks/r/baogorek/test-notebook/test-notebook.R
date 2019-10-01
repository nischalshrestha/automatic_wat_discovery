list.files("../input")

input_df <- read.csv("../input/train.csv")

train_df <- input_df[1:500, ]



avg_age <- mean(train_df$Age, na.rm = TRUE)

print(avg_age)

train_df$Age <- ifelse(is.na(train_df$Age), avg_age, train_df$Age)

summary(train_df)



val_df <- input_df[501:891, ]



test_df <- read.csv("../input/test.csv")

head(train_df)
my_glm <- glm(Survived ~ Age + Pclass + Sex + SibSp, data = train_df,

              family = "binomial")

summary(my_glm)
val_df$Age <- ifelse(is.na(val_df$Age), avg_age, val_df$Age)

pred_val <- predict(my_glm, type = "response", newdata = val_df)

hist(pred_val)

pred_survived <- as.numeric(pred_val > .5)
n_correct <- sum(pred_survived == val_df$Survived, na.rm = TRUE)

n_incorrect <- sum(pred_survived != val_df$Survived, na.rm = TRUE)

print(n_correct)

print(n_incorrect)

accuracy <- n_correct / (n_correct + n_incorrect)

print(accuracy)
pred <- predict(my_glm, type = "response")

pred_test <- predict(my_glm, newdata = test_df, type = "response")

hist(pred_test)
test_df$Age <- ifelse(is.na(test_df$Age), avg_age, test_df$Age)

test_df$Survived <- as.numeric(pred_test > .5)

test_df$Survived <- ifelse(is.na(test_df$Survived), 0, test_df$Survived)

summary(test_df)

submission_df <- test_df[, c("PassengerId", "Survived")]

write.csv(submission_df, "submission.csv", row.names = FALSE)