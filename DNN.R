# dnn
library(pROC)
library(ggplot2)

library(caret)
library(keras)
library(pROC)

scale01 <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

normalize_columns <- function(data, cols) {
  data[, cols] <- lapply(data[, cols, drop = FALSE], scale01)
  data
}

train_data_normalized <- normalize_columns(train_data, c(1, 3:29))
test_data_normalized <- normalize_columns(test_data, c(1, 3:29))

head(train_data_normalized)
head(test_data_normalized)

num_negatives <- sum(train_data_normalized$ch_ari == 0)
num_positives <- sum(train_data_normalized$ch_ari == 1)
ratio <- num_negatives / num_positives

X_train <- as.matrix(subset(train_data_normalized, select = -ch_ari))
y_train <- train_data_normalized$ch_ari
X_test <- as.matrix(subset(test_data_normalized, select = -ch_ari))
y_test <- test_data_normalized$ch_ari
input_shape <- ncol(X_train)

class_weights <- list("0" = 1, "1" = ratio)

model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = 'relu', input_shape = input_shape) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 16, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1, activation = 'sigmoid')

model %>% compile(
  optimizer = optimizer_adam(),
  loss = 'binary_crossentropy',
  metrics = 'accuracy'
)

history <- model %>% fit(
  X_train, y_train,
  epochs = 20, batch_size = 32,
  validation_split = 0.2,
  verbose = 1,
  class_weight = class_weights
)

predictions_test <- model %>% predict(X_test)
roc_curve_DNN_test <- roc(y_test, predictions_test)
auc_value_DNN_test <- auc(roc_curve_DNN_test)
predictions_binary_test <- ifelse(predictions_test > 0.5, 1, 0)
predictions_binary_test <- factor(predictions_binary_test, levels = c(0, 1))
y_test <- factor(y_test, levels = c(0, 1))
conf_matrix_test <- confusionMatrix(predictions_binary_test, y_test)

accuracy_dnn_test <- conf_matrix_test$overall['Accuracy']
precision_dnn_test <- conf_matrix_test$byClass['Precision']
recall_dnn_test <- conf_matrix_test$byClass['Sensitivity']
f1_score_dnn_test <- 2 * (precision_dnn_test * recall_dnn_test) / (precision_dnn_test + recall_dnn_test)

predictions_train <- model %>% predict(X_train)
roc_curve_DNN_train <- roc(y_train, predictions_train)
auc_value_DNN_train <- auc(roc_curve_DNN_train)
predictions_binary_train <- ifelse(predictions_train > 0.5, 1, 0)
predictions_binary_train <- factor(predictions_binary_train, levels = c(0, 1))
y_train <- factor(y_train, levels = c(0, 1))
conf_matrix_train <- confusionMatrix(predictions_binary_train, y_train)

accuracy_dnn_train <- conf_matrix_train$overall['Accuracy']
precision_dnn_train <- conf_matrix_train$byClass['Precision']
recall_dnn_train <- conf_matrix_train$byClass['Sensitivity']
f1_score_dnn_train <- 2 * (precision_dnn_train * recall_dnn_train) / (precision_dnn_train + recall_dnn_train)

print(paste("AUC:", auc_value_DNN_test))
print(paste("Accuracy:", accuracy_dnn_test))
print(paste("Precision:", precision_dnn_test))
print(paste("Recall:", recall_dnn_test))
print(paste("F1-Score:", f1_score_dnn_test))

print("Train Data Metrics:")
print(paste("AUC:", auc_value_DNN_train))
print(paste("Accuracy:", accuracy_dnn_train))
print(paste("Precision:", precision_dnn_train))
print(paste("Recall:", recall_dnn_train))
print(paste("F1-Score:", f1_score_dnn_train))







