setwd("C:/Users/carpe/Downloads")
dropout.data <- read.csv("Updated Dataset - INFO 3237.csv", sep=",", 
                 header=T, strip.white = T, na.strings = c("NA","NaN","","?"))
library(RTextTools)
library(reshape2)
library(textdata)
library(syuzhet)
library(wordcloud)
library(data.table)
library(tm)
library(topicmodels)
library(slam)
library(tidyr)
library(tidytext)
library(tidyverse)
library(dplyr)
library(plyr)
library(ggplot2)
library(topicmodels)
library(SnowballC)
library(RColorBrewer)
library(rattle)

dropout.data <- dropout.data %>% drop_na()

glimpse(dropout.data)
as.factor(dropout.data$Target)
dropout.data$Target <- ifelse(dropout.data$Target == "Dropout", 1, 0)
str(dropout.data)
View(dropout.data)

index <- createDataPartition(dropout.data$Target, times = 1, p = 0.7, 
                             list = FALSE)
train_data <- dropout.data[index, ]
validation_data <- dropout.data[-index, ]

set.seed(100)
dropout_tree <- rpart(Target ~., data = train_data)
fancyRpartPlot(dropout_tree)
summary(dropout_tree)
printcp(dropout_tree)
plotcp(dropout_tree)

dropout_prunetree <- prune(dropout_tree, cp = dropout_tree$cptable[which.min(dropout_tree$cptable[, "xerror"]), "CP"])
fancyRpartPlot(dropout_prunetree)

predicted_values <- predict(dropout_prunetree, validation_data, type = "matrix")
prediction <- factor(ifelse(predicted_values[] > 0.5, 1, 0))
dropout_cm <- confusionMatrix(prediction, as.factor(validation_data$Target),
                             positive = levels(validation_data$Target)[2])
dropout_cm

control <- trainControl(method = "cv", number = 10)
model.rf50 <- train(Target ~., 
                    data = train_data, 
                    method = "rf",
                    ntree = 50, 
                    tuneGrid = expand.grid(.mtry = 2:8),
                    trControl = control)
print(model.rf50)
plot(model.rf50)
plot(varImp(model.rf50))

predicted_values2 <- predict(model.rf50, validation_data, type = "raw")
prediction2 <- factor(ifelse(predicted_values2[] > 0.5, 1, 0))
dropout_cm2 <- confusionMatrix(prediction2, as.factor(validation_data$Target),
                               positive = levels(validation_data$Target)[2])
dropout_cm2

model.rf100 <- train(Target ~., 
                     data = train_data, 
                     method = "rf",
                     ntree = 100, 
                     tuneGrid = expand.grid(.mtry = 2:8),
                     trControl = control)
print(model.rf100)
plot(model.rf100)
plot(varImp(model.rf100))

predicted_values3 <- predict(model.rf100, validation_data, type = "raw")
prediction3 <- factor(ifelse(predicted_values3[] > 0.5, 1, 0))
dropout_cm3 <- confusionMatrix(prediction3, as.factor(validation_data$Target),
                               positive = levels(validation_data$Target)[2])
dropout_cm3

model.rf10 <- train(Target ~., 
                     data = train_data, 
                     method = "rf",
                     ntree = 10, 
                     tuneGrid = expand.grid(.mtry = 2:8),
                     trControl = control)
print(model.rf10)
plot(model.rf10)
plot(varImp(model.rf10))

predicted_values4 <- predict(model.rf10, validation_data, type = "raw")
prediction4 <- factor(ifelse(predicted_values4[] > 0.5, 1, 0))
dropout_cm4 <- confusionMatrix(prediction4, as.factor(validation_data$Target),
                               positive = levels(validation_data$Target)[2])
dropout_cm4

#ntree = 50 is most accurate (87.97%)
#changing mtry value did not improve accuracy at all

