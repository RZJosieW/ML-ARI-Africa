# LGB
library(lightgbm)
library(caret)
library(pROC)

train_data_LGB <- as.matrix(train_data[, -which(names(train_data) == "ch_ari")])
train_label_LGB <- train_data$ch_ari
test_data_LGB<- as.matrix(test_data[, -which(names(test_data) == "ch_ari")])
test_label_LGB<- test_data$ch_ari

dtrain_LGB<- lgb.Dataset(data = train_data_LGB, label = train_label_LGB)
dtest_LGB<- lgb.Dataset(data = test_data_LGB, label = test_label_LGB, free_raw_data = FALSE)

params <- list(
  objective = "binary",
  metric = "auc",
  boosting_type = "gbdt",
  learning_rate = 0.05,
  num_leaves = 64,  
  min_data_in_leaf = 30,
  bagging_fraction = 0.7,
  feature_fraction = 0.6,
  scale_pos_weight = ratio,  
  seed = 125
)

num_rounds <- 100
cv_results <- lgb.cv(
  params = params,
  data = dtrain_LGB,
  nfold = 5,
  nrounds = num_rounds,
  early_stopping_rounds = 20,
  verbose = 1)

best_nrounds <- cv_results$best_iter

final_model <- lgb.train(
  params = params,
  data = dtrain_LGB,
  nrounds = best_nrounds,
  valids = list(test = dtest_LGB),
  verbose = 1)

pred_probs_test <- predict(final_model, test_data_LGB, num_iteration = final_model$best_iter)
pred_class_test <- ifelse(pred_probs_test > 0.5, 1, 0)
conf_matrix_test <- confusionMatrix(factor(pred_class_test, levels = c(0, 1)), 
                                    factor(test_label_LGB, levels = c(0, 1)))

accuracy_lightgbm_test <- conf_matrix_test$overall['Accuracy']
precision_lightgbm_test <- conf_matrix_test$byClass['Precision']
recall_lightgbm_test <- conf_matrix_test$byClass['Sensitivity']
f1_score_lightgbm_test <- 2 * (precision_lightgbm_test * recall_lightgbm_test) / (precision_lightgbm_test + recall_lightgbm_test)

roc_obj_lightgbm_test <- roc(response = factor(test_label_LGB, levels = c(0, 1)), predictor = pred_probs_test)
auc_value_lightgbm_test <- auc(roc_obj_lightgbm_test)

pred_probs_train <- predict(final_model, train_data_LGB, num_iteration = final_model$best_iter)
pred_class_train <- ifelse(pred_probs_train > 0.5, 1, 0)
conf_matrix_train <- confusionMatrix(factor(pred_class_train, levels = c(0, 1)), 
                                     factor(train_label_LGB, levels = c(0, 1)))

accuracy_lightgbm_train <- conf_matrix_train$overall['Accuracy']
precision_lightgbm_train <- conf_matrix_train$byClass['Precision']
recall_lightgbm_train <- conf_matrix_train$byClass['Sensitivity']
f1_score_lightgbm_train <- 2 * (precision_lightgbm_train * recall_lightgbm_train) / (precision_lightgbm_train + recall_lightgbm_train)

roc_obj_lightgbm_train <- roc(response = factor(train_label_LGB, levels = c(0, 1)), predictor = pred_probs_train)
auc_value_lightgbm_train <- auc(roc_obj_lightgbm_train)

print("Test Data Metrics:")
print(paste("Accuracy:", accuracy_lightgbm_test))
print(paste("Precision:", precision_lightgbm_test))
print(paste("Recall:", recall_lightgbm_test))
print(paste("F1 Score:", f1_score_lightgbm_test))
print(paste("AUC:", auc_value_lightgbm_test))

print("Train Data Metrics:")
print(paste("Accuracy:", accuracy_lightgbm_train))
print(paste("Precision:", precision_lightgbm_train))
print(paste("Recall:", recall_lightgbm_train))
print(paste("F1 Score:", f1_score_lightgbm_train))
print(paste("AUC:", auc_value_lightgbm_train))


# catboost 
# Install and load the necessary libraries
devtools::install_github('catboost/catboost', subdir = 'catboost/R-package')
library(catboost)
library(caret)
library(pROC)

train_data_catboost <- train_data[, -which(names(train_data) == "ch_ari")]
train_label <- train_data$ch_ari
test_data_catboost <- test_data[, -which(names(test_data) == "ch_ari")]
test_label <- test_data$ch_ari
dtrain_cat <- catboost.load_pool(data = as.matrix(train_data_catboost), label = train_label)
dtest_cat <- catboost.load_pool(data = as.matrix(test_data_catboost), label = test_label)

num_negatives <- sum(train_label == 0)
num_positives <- sum(train_label == 1)
ratio <- num_negatives / num_positives
class_weights <- c(1, ratio)

num_rounds <- 100
params_catboost <- list(
  loss_function = "Logloss",
  eval_metric = "AUC",
  learning_rate = 0.05,
  depth = 6,
  min_data_in_leaf = 30,
  bagging_temperature = 0.7,
  rsm = 0.6,
  class_weights = class_weights,
  iterations = num_rounds,
  random_seed = 125
)

cv_results_cat <- catboost.cv(
  params = params_catboost,
  pool = dtrain_cat,
  fold_count = 5,
  partition_random_seed = 125,
  early_stopping_rounds = 20
)

best_nrounds_cat <- cv_results_cat$best_iteration

final_model_cat <- catboost.train(
  learn_pool = dtrain_cat,
  params = params_catboost
)

pred_prob_cat_test <- catboost.predict(final_model_cat, dtest_cat, prediction_type = "Probability")
pred_class_cat_test <- ifelse(pred_prob_cat_test > 0.5, 1, 0)
conf_matrix_test <- confusionMatrix(as.factor(pred_class_cat_test), as.factor(test_label))

accuracy_catboost_test <- conf_matrix_test$overall["Accuracy"]
recall_catboost_test <- conf_matrix_test$byClass["Recall"]
precision_catboost_test <- conf_matrix_test$byClass["Precision"]
f1_catboost_test <- conf_matrix_test$byClass["F1"]
auc_catboost_test <- roc(test_label, pred_prob_cat_test)$auc

pred_prob_cat_train <- catboost.predict(final_model_cat, dtrain_cat, prediction_type = "Probability")
pred_class_cat_train <- ifelse(pred_prob_cat_train > 0.5, 1, 0)
conf_matrix_train <- confusionMatrix(as.factor(pred_class_cat_train), as.factor(train_label))

accuracy_catboost_train <- conf_matrix_train$overall["Accuracy"]
recall_catboost_train <- conf_matrix_train$byClass["Recall"]
precision_catboost_train <- conf_matrix_train$byClass["Precision"]
f1_catboost_train <- conf_matrix_train$byClass["F1"]
auc_catboost_train <- roc(train_label, pred_prob_cat_train)$auc

print("Test Data Metrics:")
print(paste("AUC:", auc_catboost_test))
print(paste("Accuracy:", accuracy_catboost_test))
print(paste("Recall:", recall_catboost_test))
print(paste("Precision:", precision_catboost_test))
print(paste("F1-Score:", f1_catboost_test))

print("Train Data Metrics:")
print(paste("AUC:", auc_catboost_train))
print(paste("Accuracy:", accuracy_catboost_train))
print(paste("Recall:", recall_catboost_train))
print(paste("Precision:", precision_catboost_train))
print(paste("F1-Score:", f1_catboost_train))
