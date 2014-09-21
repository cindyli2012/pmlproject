training <-read.csv('pml-training.csv')
testing <-read.csv('pml-testing.csv')
set.seed(1223)

# delete columns with # of NA >=500
clean.train <- training[colSums(is.na(training)) < 500]
# remove irrelevant variables: X, user_name, raw_timestram_part_1, raw_timestram_part_2, cvtd_timestamp,
# new_window, num_window
clean.train <- clean.train[,-1:-7]
# remove near Zero Variables
clean.train <- clean.train[,!nearZeroVar(clean.train,saveMetrics=T)$nzv]

subsets <- createFolds(clean.train$classe, k = 10, list = TRUE, returnTrain = FALSE)
err.vec <- numeric(0)
fits <- list()

for (i in 1:10) {
  fits[[i]] <- randomForest(clean.train$classe ~ .,data=train,subset=subsets[[i]])
  prediction <- predict(fits[[i]],newdata=clean.train[-subsets[[i]],],type="class")
  cfMatrix <- table(clean.train$classe[-subsets[[i]]],prediction)
  print(cfMatrix)
  err <- 1.0 - sum(diag(cfMatrix))/sum(cfMatrix)
  err.vec <- rbind(err.vec,err)
}
err.vec
mean(err.vec)

#randomForest fit to all training data
finalFit <- randomForest(clean.train$classe ~ .,data=clean.train)

#prediction for testing data
prediction <- predict(finalFit,newdata=testing,type="class")

