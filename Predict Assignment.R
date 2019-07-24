'The goal of your project is to predict the manner in which they did the exercise.
This is the “classe” variable in the training set. You may use any of the other
variables to predict with. You should create a report describing how you built
your model, how you used cross validation, what you think the expected out of
sample error is, and why you made the choices you did. You will also use your
prediction model to predict 20 different test cases.'

rm(list = ls())
setwd("C:/Users/nicolelin/Desktop/Coursera_Data Science/week4")

'1 Load the dataset and briefly view the characteristics of the data'
data <- read.csv("pml-training.csv")
colnames(data)
summary(data)

'2 Use cross-validation method to built a valid model; 70% of the original data 
is used for model building (training data) while the rest of 30% of the data 
is used for testing (testing data)'
library(caret)
set.seed(1111)
split <- createDataPartition(y=data$classe, p=.70,list=F)
training <- data[split,]
testing <- data[-split,]

'3 Since the 160 variables in the training data is too large, clean the 
data by 
1) excluding variables which apparently cannot be explanatory variables, 
2) reducing variables with little information.'

#exclude identifier, timestamp, and window data (they cannot be used for prediction)
Cl <- grep("name|timestamp|window|X", colnames(training), value=F) 
trainingCl <- training[,-Cl]
#select variables with high (over 95%) missing data --> exclude them from the analysis
trainingCl[trainingCl==""] <- NA
NArate <- apply(trainingCl, 2, function(x) sum(is.na(x)))/nrow(trainingCl)
trainingCl <- trainingCl[!(NArate>0.95)]
summary(trainingCl)

'4 Apply PCA to reduce the number of variables'
# preProc <- preProcess(trainingCl[,1:52], method="pca",thresh=.8) #12 components are required
# preProc <- preProcess(trainingCl[,1:52], method="pca",thresh=.9) #18 components are required
# preProc <- preProcess(trainingCl[,1:52], method="pca",thresh=.95) #25 components are required

preProc <- preProcess(trainingCl[,1:52],method="pca",pcaComp=25) 
preProc$rotation
trainingPC <- predict(preProc,trainingCl[,1:52])

'5 Apply random forest method to build a model'
library(randomForest)
modFitRF <- randomForest(trainingCl$classe ~ .,   
                         data=trainingPC, do.trace=F)
print(modFitRF) # view results 
importance(modFitRF) # importance of each predictor

'6 Check the model with the testing data set'
testingCl <- testing[,-Cl]
testingCl[testingCl==""] <- NA
NArate <- apply(testingCl, 2, function(x) sum(is.na(x)))/nrow(testingCl)
testingCl <- testingCl[!(NArate>0.95)]
testingPC <- predict(preProc,testingCl[,1:52])
confusionMatrix(testingCl$classe,predict(modFitRF,testingPC))

'7 Apply the model to estimate classes of 20 observations'
testdata <- read.csv("pml-testing.csv")
testdataCl <- testdata[,-Cl]
testdataCl[testdataCl==""] <- NA
NArate <- apply(testdataCl, 2, function(x) sum(is.na(x)))/nrow(testdataCl)
testdataCl <- testdataCl[!(NArate>0.95)]
testdataPC <- predict(preProc,testdataCl[,1:52])
testdataCl$classe <- predict(modFitRF,testdataPC)
