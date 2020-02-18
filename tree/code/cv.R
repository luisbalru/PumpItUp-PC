library(RWeka)
library(caret)

dataset <- read.csv("../train-preprocessed.csv")
dataset$X <- NULL
dataset$id <- NULL
folds <- createFolds(dataset$status_group, k=5, list=T)

out <- sapply(1:5, function(x) {
    train <- dataset[-folds[[x]],]
    test <- dataset[folds[[x]],]

    fit <- J48(status_group ~ ., data=train)

    predictions <- predict(fit, test)

    sum(predictions == test$status_group)/length(predictions)
})

mean(out)
