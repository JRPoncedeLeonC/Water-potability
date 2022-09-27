#Installing necessary packages
if(!require(tidyverse)) install.packages("tidyverse", 
                                         repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", 
                                     repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", 
                                          repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", 
                                       repos = "http://cran.us.r-project.org")
if(!require(naivebayes)) install.packages("naivebayes", 
                                          repos = "http://cran.us.r-project.org")
if(!require(gbm)) install.packages("gbm", 
                                   repos = "http://cran.us.r-project.org")
if(!require(kernlab)) install.packages("kernlab", 
                                   repos = "http://cran.us.r-project.org")

#LIBRARY IMPORT 
library(tidyverse)
library(caret)
library(data.table)
library(xgboost)
library(naivebayes)
library(gbm)
library(kernlab)

#DATA IMPORT
wp <- read.csv("water_potability.csv")

#DATA SPLITTING
y <- wp$Potability
set.seed(42)
test_index <- createDataPartition(y, times=1, p=0.2, list=F)
train_data <- wp[-test_index,]
test_data <- wp[test_index,]

#DATA CLEANING
ReplaceNa <- function(df){ #function to replace the missing values
  df_1 <- df %>% 
    replace_na(list(ph=mean(df$ph,na.rm=T),
                    Sulfate=mean(df$Sulfate,na.rm=T),
                    Trihalomethanes=mean(df$Trihalomethanes,na.rm=T)))
  return(df_1)
}
ScaleData <- function(df){ #function to scale the data
  df_pred <- df %>% select(-Potability) %>% scale() 
  df_y <- df %>% select(Potability) 
  return(cbind(df_pred, df_y))
}
#data sets 1
train_data_1 <- ReplaceNa(train_data) %>% ScaleData #clean data with NAs as mean and scaled 
test_data_1 <- ReplaceNa(test_data) %>% ScaleData #clean data with NAs as mean and scaled  
#data sets 2
train_data_2 <- train_data[complete.cases(train_data),] %>% ScaleData() #scaled clean data without NA rows
test_data_2 <- test_data[complete.cases(test_data),] %>% ScaleData() #scaled clean data without NA rows

#MODEL TRAINING AND EVALUATION
train_control <- trainControl(method = "cv", number = 10) #for algorithms to perform cross validation
#dataframe of results
res <- data.frame(matrix(nrow = 0, ncol = 4))
colnames(res) <- c("model", "data_set", "accuracy", "f_1")

#* KNN + data sets 1
set.seed(42)
train_knn <- train(factor(Potability) ~ ., method = "knn",
                   tuneGrid = data.frame(k = seq(3, 45, 2)),
                   data = train_data_1, 
                   trControl = train_control)
pred_knn <- predict(train_knn, test_data_1)
#data stored
res[1,] <- c("KNN", 1, as.numeric(confusionMatrix(pred_knn,
                                                  factor(test_data_1$Potability))$overall["Accuracy"]),
             F_meas(data = pred_knn, reference = factor(test_data_1$Potability)))

#* Random Forest + data sets 1
set.seed(42)
train_rf <- train(factor(Potability) ~., method = "rf",
                  data = train_data_1,
                  tuneGrid = data.frame(mtry = seq(3:15)), #parameter tuning
                  trControl = train_control, #cross-validation
                  ntree = 200) #number of trees
pred_rf <- predict(train_rf, test_data_1)
#we store the data + data sets 1
res[nrow(res)+1,] <- c("rf", 1, as.numeric(confusionMatrix(pred_rf,
                                                           factor(test_data_1$Potability))$overall["Accuracy"]),
                       F_meas(data = pred_rf, reference = factor(test_data_1$Potability)))

#* Logistic Regression + data sets 1
set.seed(42)
train_glm <- train(factor(Potability) ~., method = "glm",
                   data = train_data_1)
pred_glm <- predict(train_glm, test_data_1)
res[nrow(res)+1,] <- c("glm", 1, as.numeric(confusionMatrix(pred_glm,
                                                            factor(test_data_1$Potability))$overall["Accuracy"]),
                       F_meas(data = pred_glm, reference = factor(test_data_1$Potability)))

#* XGBoost + data sets 1
train_control1 <- trainControl(method = "cv", number = 3)
set.seed(42)
train_xgb <- train(factor(Potability) ~., method = "xgbDART",
                   data = train_data_1,
                   trControl = train_control1,
                   tuneGrid = expand.grid(
                     nrounds = c(11),
                     max_depth = 8,#c(6, 7, 8)
                     eta = 0.01, #c(0, 0.01, 0.1, 0.2)
                     gamma = 0.01, #c(0, 0.001, 0.01)
                     subsample = 0.5,
                     colsample_bytree = 1,#c(0.5, 0.8, 1)
                     rate_drop = 0, #c(0, 0.4,0.5)
                     skip_drop = 0.5, #c(0, 0.4,0.5)
                     min_child_weight = 1 #c(0,1,2)
                   ))
pred_xgb <- predict(train_xgb, test_data_1)
res[nrow(res)+1,] <- c("xgbDART", 1, as.numeric(confusionMatrix(pred_xgb,
                                                                factor(test_data_1$Potability))$overall["Accuracy"]),
                       F_meas(data = pred_xgb, reference = factor(test_data_1$Potability)))

#* Naive Bayes + data sets 1:
set.seed(42)
train_nb <- train(factor(Potability) ~., data = train_data_1, method = "naive_bayes")
pred_nb <- predict(train_nb, test_data_1)
res[nrow(res)+1,] <- c("naive_bayes", 1, as.numeric(confusionMatrix(pred_nb,
                                                                    factor(test_data_1$Potability))$overall["Accuracy"]),
                       F_meas(data = pred_nb, reference = factor(test_data_1$Potability)))

#* xgbTree + data sets 1:
set.seed(42)
train_xgbT <- train(factor(Potability) ~., data = train_data_1, 
                    method = "xgbTree",
                    trControl = train_control,
                    tuneGrid = expand.grid(
                      nrounds = 65, #c(63,65,70)
                      eta=0.05, #c(0.01, 0.05)
                      max_depth=6,#c(6, 7, 8)
                      colsample_bytree=1, #c(0.5, 1)
                      subsample=0.5, 
                      gamma=0.01, #c(0.0001, 0.001, 0.01)
                      min_child_weight=0)) #c(0,1,2)
pred_xgbT <- predict(train_xgbT, test_data_1)
res[nrow(res)+1,] <- c("xgbTree", 1, as.numeric(confusionMatrix(pred_xgbT,
                                                                factor(test_data_1$Potability))$overall["Accuracy"]),
                       F_meas(data = pred_xgbT, reference = factor(test_data_1$Potability)))

#* QDA + data sets 1:
set.seed(42)
train_qda <- train(factor(Potability) ~., data = train_data_1, 
                   method = "qda")
pred_qda <- predict(train_qda, test_data_1)
res[nrow(res)+1,] <- c("qda", 1, as.numeric(confusionMatrix(pred_qda,
                                                            factor(test_data_1$Potability))$overall["Accuracy"]),
                       F_meas(data = pred_qda, reference = factor(test_data_1$Potability)))

#* Least Squares Support Vector Machine + data sets 1:
set.seed(42)
train_svm <- train(factor(Potability) ~., method = "lssvmRadial",
                   data = train_data_1,
                   tuneGrid = expand.grid(
                     tau = 0.009,#c(0.001,0.009,0.01,0.1)	
                     sigma = 0.01#c(0.001,0.009,0.01,0.1)	
                   ), 
                   trControl = train_control) 
pred_svm <- predict(train_svm, test_data_1)
res[nrow(res)+1,] <- c("lssvmRadial", 1, as.numeric(confusionMatrix(pred_svm,
                                                                    factor(test_data_1$Potability))$overall["Accuracy"]),
                       F_meas(data = pred_svm, reference = factor(test_data_1$Potability)))

#* Least Squares Support Vector Machine + data sets 2:
set.seed(42)
train_svm1 <- train(factor(Potability) ~., method = "lssvmRadial",
                    data = train_data_2,
                    tuneGrid = expand.grid(
                      tau = 0.1, #c(0.001,0.009,0.01,0.1)
                      sigma = 0.01 #c(0.001,0.009,0.01,0.1)
                    ), 
                    trControl = train_control) 
pred_svm1 <- predict(train_svm1, test_data_2)
res[nrow(res)+1,] <- c("lssvmRadial", 2, as.numeric(confusionMatrix(pred_svm1,
                                                                    factor(test_data_2$Potability))$overall["Accuracy"]),
                       F_meas(data = pred_svm1, reference = factor(test_data_2$Potability)))

#* Random Forest + data sets 2:
set.seed(42)
train_rf1 <- train(factor(Potability) ~., method = "rf",
                   data = train_data_2,
                   tuneGrid = data.frame(mtry = seq(3:15)), #parameter tuning mtry = 5
                   trControl = train_control, #cross-validation
                   ntree = 200) #number of trees
pred_rf1 <- predict(train_rf1, 
                    test_data_2)
res[nrow(res)+1,] <- c("rf", 2, as.numeric(confusionMatrix(pred_rf1,
                                                           factor(test_data_2$Potability))$overall["Accuracy"]),
                       F_meas(data = pred_rf1, reference = factor(test_data_2$Potability)))

#* xgbTree + data sets 2:
set.seed(42)
train_xgbT1 <- train(factor(Potability) ~., data = train_data_2, 
                     method = "xgbTree",
                     trControl = train_control,
                     tuneGrid = expand.grid(
                       nrounds=65, # c(63,65,70)
                       eta=0.05, #c(0.01, 0.05)
                       max_depth=6,#c(6, 7, 8)
                       colsample_bytree=1, #c(0.5, 1)
                       subsample=0.5, 
                       gamma=0.01, #c(0.0001, 0.001, 0.01)
                       min_child_weight=0)) #c(0,1,2)
pred_xgbT1 <- predict(train_xgbT1, test_data_2)
res[nrow(res)+1,] <- c("xgbTree", 2, as.numeric(confusionMatrix(pred_xgbT1,
                                                                factor(test_data_2$Potability))$overall["Accuracy"]),
                       F_meas(data = pred_xgbT1, reference = factor(test_data_2$Potability)))

#RESULTS:
res %>% ggplot(aes(as.numeric(accuracy), 
                   as.numeric(f_1), col = model, shape = data_set)) + 
  geom_point() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("Accuracy") + ylab("F1 score")

res %>% filter(accuracy == max(accuracy) & f_1 == max(f_1))

#DATA EXPORT
write.csv(wp,"water_potability_export.csv", row.names = T)















