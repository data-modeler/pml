# The data for this project come from this source: 
#   http://groupware.les.inf.puc-rio.br/har. 
# If you use the document you create for this class 
# for any purpose please cite them as they have been 
# very generous in allowing their data to be used for 
# this kind of assignment.
rm(list=ls())
setwd("C:/Users/Dale/Dropbox/Certifications/Johns Hopkins Data Science/Practical Machine Learning")
train <- read.csv("pml-training.csv")
test <- read.csv("pml-testing.csv")
str(train)
