# Lasso
library(glmnet)
library(caret)
library(pROC)

x_train <- model.matrix(ch_ari ~ . - 1, data = train_data_standardized)
x_test <- model.matrix(ch_ari ~ . - 1, data = test_data_standardized)

y_train <- train_data_standardized$ch_ari
y_test <- test_data_standardized$ch_ari

num_negatives <- sum(y_train == 0)
num_positives <- sum(y_train == 1)
weight_for_negatives <- 1 / num_negatives * num_positives
weight_for_positives <- 1
weights <- ifelse(y_train == 1, weight_for_positives, weight_for_negatives)

cv_lasso_model <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1, weights = weights, nfolds = 5)
best_lam <- cv_lasso_model$lambda.min

lasso_best <- glmnet(x_train, y_train, alpha = 1, lambda = best_lam, weights = weights)

pred_train <- predict(lasso_best, s = best_lam, newx = x_train, type = "response")
pred_class_train <- ifelse(pred_train > 0.5, 1, 0)

y_train <- factor(y_train, levels = c(0, 1))
pred_class_train <- factor(pred_class_train, levels = c(0, 1))

conf_matrix_train <- confusionMatrix(pred_class_train, y_train)

accuracy_train <- conf_matrix_train$overall["Accuracy"]
recall_train <- conf_matrix_train$byClass["Sensitivity"]
precision_train <- conf_matrix_train$byClass["Precision"]
f1_score_train <- 2 * (precision_train * recall_train) / (precision_train + recall_train)
specificity_train <- conf_matrix_train$byClass["Specificity"]

print(paste("Training Accuracy:", accuracy_train))
print(paste("Training Recall/Sensitivity:", recall_train))
print(paste("Training Precision:", precision_train))
print(paste("Training F1 Score:", f1_score_train))
print(paste("Training Specificity:", specificity_train))

roc_lasso_train <- roc(response = y_train, predictor = as.vector(pred_train))
auc_lasso_train <- auc(roc_lasso_train)
print(paste("Training AUC:", auc_lasso_train))

pred_test <- predict(lasso_best, s = best_lam, newx = x_test, type = "response")
pred_class_test <- ifelse(pred_test > 0.5, 1, 0)

y_test <- factor(y_test, levels = c(0, 1))
pred_class_test <- factor(pred_class_test, levels = c(0, 1))

conf_matrix_test <- confusionMatrix(pred_class_test, y_test)

accuracy_test <- conf_matrix_test$overall["Accuracy"]
recall_test <- conf_matrix_test$byClass["Sensitivity"]
precision_test <- conf_matrix_test$byClass["Precision"]
f1_score_test <- 2 * (precision_test * recall_test) / (precision_test + recall_test)
specificity_test <- conf_matrix_test$byClass["Specificity"]

print(paste("Test Accuracy:", accuracy_test))
print(paste("Test Recall/Sensitivity:", recall_test))
print(paste("Test Precision:", precision_test))
print(paste("Test F1 Score:", f1_score_test))
print(paste("Test Specificity:", specificity_test))

roc_lasso_test <- roc(response = y_test, predictor = as.vector(pred_test))
auc_lasso_test <- auc(roc_lasso_test)
print(paste("Test AUC:", auc_lasso_test))



