#GB
library(caret)
library(e1071)
library(pROC)

x_train <- model.matrix(ch_ari ~ . - 1, data = train_data_standardized)
x_test <- model.matrix(ch_ari ~ . - 1, data = test_data_standardized)

y_train <- train_data_standardized$ch_ari
y_test <- test_data_standardized$ch_ari

y_train <- factor(y_train, levels = c(0, 1), labels = c("Class0", "Class1"))
y_test <- factor(y_test, levels = c(0, 1), labels = c("Class0", "Class1"))

num_negatives <- sum(y_train == "Class0")
num_positives <- sum(y_train == "Class1")
weight_for_negatives <- num_positives / num_negatives
weight_for_positives <- 1

weights <- ifelse(y_train == "Class1", weight_for_positives, weight_for_negatives)

weighted_naive_bayes <- function(x, y, w) {
  model <- naiveBayes(x, y, laplace = 1)
  return(model)
}

set.seed(123)
folds <- createFolds(y_train, k = 5, list = TRUE, returnTrain = TRUE)

results <- list()

for (i in 1:length(folds)) {
  train_indices <- folds[[i]]
  val_indices <- setdiff(seq_len(nrow(x_train)), train_indices)
  
  x_train_fold <- x_train[train_indices, ]
  y_train_fold <- y_train[train_indices]
  weights_fold <- weights[train_indices]
  
  x_val_fold <- x_train[val_indices, ]
  y_val_fold <- y_train[val_indices]
  
  model <- weighted_naive_bayes(x_train_fold, y_train_fold, weights_fold)
  
  pred_val_prob <- predict(model, newdata = x_val_fold, type = "raw")[, "Class1"]
  pred_class_val <- ifelse(pred_val_prob > 0.5, "Class1", "Class0")
  
  conf_matrix_val <- confusionMatrix(factor(pred_class_val, levels = c("Class0", "Class1")), y_val_fold)
  
  accuracy_val <- conf_matrix_val$overall["Accuracy"]
  recall_val <- conf_matrix_val$byClass["Sensitivity"]
  precision_val <- conf_matrix_val$byClass["Precision"]
  f1_score_val <- 2 * (precision_val * recall_val) / (precision_val + recall_val)
  specificity_val <- conf_matrix_val$byClass["Specificity"]
  
  roc_val <- roc(response = y_val_fold, predictor = as.vector(pred_val_prob))
  auc_val <- auc(roc_val)
  
  results[[i]] <- list(
    accuracy = accuracy_val,
    recall = recall_val,
    precision = precision_val,
    f1_score = f1_score_val,
    specificity = specificity_val,
    auc = auc_val
  )
}

# Aggregate results
agg_results <- sapply(results, function(res) unlist(res))
agg_results <- rowMeans(agg_results)



final_model <- weighted_naive_bayes(x_train, y_train, weights)

pred_train_prob <- predict(final_model, newdata = x_train, type = "raw")[, "Class1"]
pred_class_train <- ifelse(pred_train_prob > 0.5, "Class1", "Class0")
pred_class_train <- factor(pred_class_train, levels = c("Class0", "Class1"))

conf_matrix_train <- confusionMatrix(pred_class_train, y_train)

accuracy_train <- conf_matrix_train$overall["Accuracy"]
recall_train <- conf_matrix_train$byClass["Sensitivity"]
precision_train <- conf_matrix_train$byClass["Precision"]
f1_score_train <- 2 * (precision_train * recall_train) / (precision_train + recall_train)
specificity_train <- conf_matrix_train$byClass["Specificity"]

roc_gnb_train <- roc(response = y_train, predictor = as.vector(pred_train_prob))
auc_gnb_train <- auc(roc_gnb_train)

print(paste("Training Accuracy:", accuracy_train))
print(paste("Training Recall/Sensitivity:", recall_train))
print(paste("Training Precision:", precision_train))
print(paste("Training F1 Score:", f1_score_train))
print(paste("Training Specificity:", specificity_train))
print(paste("Training AUC:", auc_gnb_train))

pred_test_prob <- predict(final_model, newdata = x_test, type = "raw")[, "Class1"]
pred_class_test <- ifelse(pred_test_prob > 0.5, "Class1", "Class0")
pred_class_test <- factor(pred_class_test, levels = c("Class0", "Class1"))

conf_matrix_test <- confusionMatrix(pred_class_test, y_test)

accuracy_test <- conf_matrix_test$overall["Accuracy"]
recall_test <- conf_matrix_test$byClass["Sensitivity"]
precision_test <- conf_matrix_test$byClass["Precision"]
f1_score_test <- 2 * (precision_test * recall_test) / (precision_test + recall_test)
specificity_test <- conf_matrix_test$byClass["Specificity"]

roc_gnb_test <- roc(response = y_test, predictor = as.vector(pred_test_prob))
auc_gnb_test <- auc(roc_gnb_test)

print(paste("Test Accuracy:", accuracy_test))
print(paste("Test Recall/Sensitivity:", recall_test))
print(paste("Test Precision:", precision_test))
print(paste("Test F1 Score:", f1_score_test))
print(paste("Test Specificity:", specificity_test))
print(paste("Test AUC:", auc_gnb_test))