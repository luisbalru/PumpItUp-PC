library(RWeka)

train <- read.csv("../train-preprocessed.csv")
test <- read.csv("../test-preprocessed.csv")

fit <- J48(status_group ~ . - id, data=train)

predictions <- predict(fit, test)

merge <- data.frame(id=test$id, status_group=predictions)

write.csv(merge, "submission.csv", row.names=F, quote=F)
