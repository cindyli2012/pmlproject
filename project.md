



## 1 Problem analysis

This is a classification problem. Random forest is known to produce good accuracy for classication problem. So I focus on training with random forest.
## 2 Building model
First tried to run randomForest, which doesn’t take missing data and can not handle categorical predictors with more than 53 categories. So used the caret package with train, but it takes around 2 hours to finish. Too slow, and the fitted model can’t be used for newdata which also contains missing values.	>modFit <- train(training$classe~., data=training, method='rf',prox=TRUE)	>modFit	Random Forest 	19622 samples  	159 predictor    	5 classes: 'A', 'B', 'C', 'D', 'E' 	No pre-processing	Resampling: Bootstrapped (25 reps) 	Summary of sample sizes: 406, 406, 406, 406, 406, 406, ... 	Resampling results across tuning parameters:  	mtry  Accuracy  Kappa  Accuracy SD  Kappa SD     2  0.254     0.000  0.04562      0.00000    	117  0.861     0.824  0.04437      0.05568   	6952  0.992     0.990  0.00788      0.00998 	Accuracy was used to select the optimal model using  the largest value.	The final value used for the model was mtry = 6952.

But the test data also contains missing data, getting error like this when use the fitted model to predict.

	>prediction <- predict(modFit$finalModel, newdata=testing, na.rm=TRUE, type='response')	Error in predict.randomForest(modFit$finalModel, newdata = testing, na.rm = TRUE,  :   	missing values in newdata
So first we need to clean up the data. 
### 2.1 Clean up data
First step is to clean up the training data. The given data consists lots of NA or missing values, which need to be cleaned up. Try removing the NA values first. If generating good enough accuracy, which means they are not very necessary for prediction, we don’t need to proceed with the imputation method, which may not produce better result either. 	#delete columns with # of NA >=500
	>clean.train <- training[colSums(is.na(training)) < 500]	# remove irrelevant variables: X, user_name, raw_timestram_part_1, raw_timestram_part_2, cvtd_timestamp,	# new_window, num_window	>clean.train <- clean.train[,-1:-7]	# remove near Zero Variables	>clean.train <- clean.train[,!nearZeroVar(clean.train,saveMetrics=T)$nzv]

After the preprocessing above, the number of predictors reduced from 160 to 53. 
###2.2 Cross validation
First let’s do the cross validation for the random forest model. Here I chose the commonly used 10-fold cross validation. 

	subsets <- createFolds(clean.train$classe, k = 10, list = TRUE, returnTrain = FALSE)	err.vec <- numeric(0)	fits <- list()	for (i in 1:10) {  	fits[[i]] <- randomForest(clean.train$classe ~ .,data=train,subset=subsets[[i]])  	prediction <- predict(fits[[i]],newdata=clean.train[-subsets[[i]],],type="class")  	cfMatrix <- table(clean.train$classe[-subsets[[i]]],prediction)  	print(cfMatrix)  	err <- 1.0 - sum(diag(cfMatrix))/sum(cfMatrix)  	err.vec <- rbind(err.vec,err)	}	err.vec	mean(err.vec)	prediction       A    B    C    D    E  	A 4887   32   25   59   19  	B  214 3066  134    1    2  	C    0  154 2858   65    3  	D   11    2  110 2756   15  	E    7    9   23   19 3189   	prediction       A    B    C    D    E  	A 4990   19    1    4    8  	B  168 3069  175    3    2  	C    2  101 2943   32    2  	D    6    2  167 2707   12  	E    4   14   35   25 3168  	 prediction       A    B    C    D    E  	A 4954   22   13   28    5  	B  191 3114  104    7    1  	C    2   90 2971   17    0  	D   13    0  188 2692    1  	E    4   39   53   53 3097   	prediction       A    B    C    D    E  	A 4971   22   19    9    1  	B  162 3150   94    4    8  	C    0  157 2881   18   24  	D   42    6  182 2607   58  	E    6   20   56   20 3145   	prediction       A    B    C    D    E  	A 4965   27   26    1    3  	B  221 3037  158    1    0  	C    7  121 2922   29    0 	D   12    0  173 2689   21 	E    0   24   46   42 3134   	prediction       A    B    C    D    E  	A 4947   31   21   20    3  	B  176 3099  111   20   11 	C    6  141 2893   40    0  	D   26    2  180 2676   11  	E    3   16   82   28 3117   	prediction       A    B    C    D    E  	A 4930   34   16   37    5  	B  180 3076  141   17    3  	C    4   94 2962   20    0  	D   13    3  253 2619    6  	E    4   35   85   28 3094   	prediction       A    B    C    D    E  	A 4953   23   21   21    4  	B  154 3118  138    6    1  	C   13  102 2915   50    0  	D   45   12  192 2622   23  	E    2   19   18   23 3185   	prediction       A    B    C    D    E  	A 4932   21   28   21   20  	B  150 3169   81   12    6  	C    0  103 2933   38    5  	D    0    3  134 2738   20  	E    5   44   46   32 3119   	prediction       A    B    C    D    E  	A 4908   35   13   59    7  	B  114 3171   97   19   17  	C    6  127 2911   27    9  	D   19   15  184 2664   12  	E    4   14   48   37 3143	> err.vec    	      [,1]	err 0.05118913	err 0.04428337	err 0.04705816	err 0.05140981	err 0.05164505	err 0.05254813	err 0.05538252	err 0.04909400	err 0.04354473	err 0.04886750	> mean(err.vec)	[1] 0.04950224

From the result above, we see that the expected out of sample error is 1-0.04950224 which is slightly above 95%.  

### 2.3 Final training
The cross validation above shows random forest model is a relatively good model for the data. Then we do the final fit to all the training data to get a final model, which will be used to predict the testing data.	#randomForest fit to all training data	>finalFit <- randomForest(clean.train$classe ~ .,data=clean.train)
###2.4 Summary 
In this project, I chose random forest model based on the known fact that it outperforms many other classification approaches and original paper[2]. The accuracy validated in step 2.2 shows that it can gain a pretty good 95% accuracy in predicting data. 
## 3 Prediction on test data
Using the final model trained in 2.3, we predict the class for the testing data. The result is stored in prediction.Rdata

	#prediction for testing data	>prediction <- predict(finalFit,newdata=testing,type="class")	#save data	>save(prediction,file='prediction.Rdata')

### Note:	
The project is at Github repo:https://github.com/cindyli2012/pmlproject### [References][1] Data source: http://groupware.les.inf.puc-rio.br/har[2] Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.	
