# The data for this project come from this source: 
#   http://groupware.les.inf.puc-rio.br/har. 
# If you use the document you create for this class 
# for any purpose please cite them as they have been 
# very generous in allowing their data to be used for 
# this kind of assignment.
rm(list=ls())
library(caret)
set.seed(1016)
train <- read.csv("pml-training.csv")
test <- read.csv("pml-testing.csv")
dim(train);names(train)

# Remove unnecessary variables from train and test
# * X is not a variable but a residual row index from the csv files
# * username removed because it cannot extrapolate to other users
# * timestamps do not appear to be needed
# * new window and window index variables
# * Other variables that are merely summary statistics for the window
rm.cols <- c(1:7,12:36,50:59,69:83,87:101,103:112,125:139,141:150)
train <- train[,-rm.cols]
test <- test[,-rm.cols]
dim(train)

featurePlot(x=train[,1:5],y=train$classe,plot="pairs")
featurePlot(x=train[,6:10],y=train$classe,plot="pairs")

allmeans <- data.frame(var=names(train[,-53]),A=NA,B=NA,C=NA,D=NA,E=NA)
for (i in 1:nrow(allmeans)){
  form <- as.formula(paste0(allmeans[i,1],"~classe-1"))
  allmeans[i,2:6] <- round(lm(form,data=train)$coefficients,3)
}
allmeans

inTrain <- createDataPartition(y=train$classe,p=0.90,list=F)
mytrain <- train[inTrain,]
myvaldt <- train[-inTrain,]
dim(mytrain);dim(myvaldt)

library(parallel)
library(doParallel)

# baseline model
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
mod0 <- train(classe~.,data=mytrain, method="rf")
stopCluster(cluster)
registerDoSEQ()
table(predict(mod0$finalModel,myvaldt),myvaldt$classe)



k = 10 #number of folds in k-fold cross validation
reps = 3 #number of repeats in repeatedcv
set.seed(1234)
seeds =  as.list(sample(1:99999,1+reps*k));
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
mod1 <- train(classe~., data=mytrain, method="rf", metric="Accuracy",
              tuneGrid=expand.grid(mtry=7),  # try 7 variables as candidate splits each time
              trControl=trainControl(
                method="repeatedcv", # specifies k-fold cross validation
                number=k,   # number of folds
                repeats=reps,    # number of repeats in k-fold cv
                seeds = seeds, #to ensure reproducibility when parallel is used
                allowParallel = TRUE)
)
stopCluster(cluster)
registerDoSEQ()
mod1 # Accuracy on training data = 99.58%

predVal <- predict(mod1$finalModel,myvaldt)
table(predVal,myvaldt$classe)
valAcc <- mean(predVal==myvaldt$classe)
valAcc #Accuracy on validation data = 99.54%

predTest <- predict(mod1$finalModel,

