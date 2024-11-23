#XGBOOST
library(xgboost)
library(lime)

train_data_xgboost <- as.matrix(train_data [, -which(names(train_data ) == "ch_ari")])
train_label <- train_data $ch_ari
test_data_xgboost <- as.matrix(test_data[, -which(names(test_data) == "ch_ari")])
test_label <- test_data$ch_ari

dtrain <- xgb.DMatrix(data = train_data_xgboost, label = train_label)
dtest <- xgb.DMatrix(data = test_data_xgboost, label = test_label)

num_negatives <- sum(train_label == 0)
num_positives <- sum(train_label == 1)
ratio <- num_negatives / num_positives

params <- list(
  booster = "gbtree",
  eta = 0.05,
  max_depth = 6,
  min_child_weight = 30,
  subsample = 0.7,
  colsample_bytree = 0.6,
  objective = "binary:logistic",
  eval_metric = "auc",
  scale_pos_weight = ratio
)

num_rounds <- 100  
cv_results <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = num_rounds,
  nfold = 5,
  metrics = "auc",
  early_stopping_rounds = 20,
  verbose_eval = TRUE,
  seed = 125
)
best_nrounds <- cv_results$best_iteration

final_model_xgboost<- xgb.train(
  params = params,
  data = dtrain,
  nrounds = best_nrounds,
  watchlist = list(train = dtrain, test = dtest),
  verbose = 1
)

pred_probs <- predict(final_model_xgboost, dtest)
pred_class <- ifelse(pred_probs > 0.5, 1, 0)
library(pROC)
conf_matrix <- confusionMatrix(factor(pred_class, levels = c(0, 1)), 
                               factor(test_label, levels = c(0, 1)))

accuracy_xgboost <- conf_matrix$overall['Accuracy']
precision_xgboost <- conf_matrix$byClass['Precision']
recall_xgboost <- conf_matrix$byClass['Sensitivity']


f1_score_xgboost <- 2 * (precision_xgboost * recall_xgboost) / (precision_xgboost + recall_xgboost)
print(accuracy_xgboost)
print(precision_xgboost)
print(recall_xgboost)
print(f1_score_xgboost)
roc_obj_xgboost <- roc(response = factor(test_label, levels = c(0, 1)), predictor = pred_probs)
auc_value_xgboost <- auc(roc_obj_xgboost)
print(auc_value_xgboost)

pred_probs_train <- predict(final_model_xgboost, dtrain)

pred_class_train <- ifelse(pred_probs_train > 0.5, 1, 0)

conf_matrix_train <- confusionMatrix(factor(pred_class_train, levels = c(0, 1)), 
                                     factor(train_label, levels = c(0, 1)))

accuracy_xgboost_train <- conf_matrix_train$overall['Accuracy']
precision_xgboost_train <- conf_matrix_train$byClass['Precision']
recall_xgboost_train <- conf_matrix_train$byClass['Sensitivity']
f1_score_xgboost_train <- 2 * (precision_xgboost_train * recall_xgboost_train) / (precision_xgboost_train + recall_xgboost_train)

print(accuracy_xgboost_train)
print(precision_xgboost_train)
print(recall_xgboost_train)
print(f1_score_xgboost_train)
roc_obj_xgboost_train<- roc(response = factor(train_label, levels = c(0, 1)), predictor = pred_probs)

library(SHAPforxgboost)
shap_long <- shap.prep(xgb_model = final_model_xgboost, X_train = train_data_xgboost)
shap.plot.summary(shap_long)
shap.plot.dependence(data_long = shap_long, x = "LONGNUM")
shap.plot.dependence(data_long = shap_long, x = "age_of_child_in_years")

