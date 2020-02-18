library(RWeka)

## Reading of preprocessed data
train <- read.csv("train-preprocessed.csv")
test <- read.csv("test-preprocessed.csv")

## Fitting
fit <- J48(status_group ~ . - id, data=train)

## Prediction of test
predictions <- predict(fit, test)

## Data frame with submission format
merge <- data.frame(id=test$id, status_group=predictions)

## Save to memory
write.csv(merge, "submission.csv", row.names=F, quote=F)
