



## 1 Problem analysis

This is a classification problem. 
## 2 Building model
First tried to run randomForest, which doesn’t take missing data and can not handle categorical predictors with more than 53 categories. 

But the test data also contains missing data, getting error like this when use the fitted model to predict.

	>prediction <- predict(modFit$finalModel, newdata=testing, na.rm=TRUE, type='response')

### 2.1 Clean up data
First step is to clean up the training data. The given data consists lots of NA or missing values, which need to be cleaned up. 
	>clean.train <- training[colSums(is.na(training)) < 500]

After the preprocessing above, the number of predictors reduced from 160 to 53. 
###2.2 Cross validation
First let’s do the cross validation for the random forest model. Here I chose the commonly used 10-fold cross validation. 

	subsets <- createFolds(clean.train$classe, k = 10, list = TRUE, returnTrain = FALSE)

From the result above, we see that the expected out of sample error is 1-0.04950224 which is slightly above 95%.  

### 2.3 Final training
The cross validation above shows random forest model is a relatively good model for the data. Then we do the final 
###2.4 Summary 
In this project, I chose random forest model based on the known fact that it outperforms many other classification approaches and original paper[2]. The accuracy validated in step 2.2 shows that it can gain a pretty good 95% accuracy in predicting data. 
## 3 Prediction on test data
Using the final model trained in 2.3, we predict the class for the testing data. The result is stored in prediction.Rdata

	#prediction for testing data

### Note:
The project is at Github repo: