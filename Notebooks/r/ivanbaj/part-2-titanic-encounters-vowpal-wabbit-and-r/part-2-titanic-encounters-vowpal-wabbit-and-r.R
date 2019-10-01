library(dplyr, warn.conflicts = FALSE)



test_tbl <- tbl_df(read.csv('../input/test.csv', stringsAsFactors = FALSE))



submit <- tbl_df(select(test_tbl,PassengerId))

submit <- mutate(submit, Survived = 0)

write.csv(submit,file = "submit_all_die.csv", row.names = F)
submit <- tbl_df(select(test_tbl,PassengerId))

submit <- mutate(submit, Survived = 1)

write.csv(submit,file = "submit_all_survive.csv", row.names = F)