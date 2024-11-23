# Load data
roc_data_dnn <- read.csv("roc_data_dnn.csv")
auc_value_dnn <- read.csv("auc_value_dnn.csv")

dnn_roc_df <- data.frame(
  TPR = roc_data_dnn$TPR,
  FPR = roc_data_dnn$FPR,
  Method = "Deep Neural Network"
)

rocobjs <- list(
  Naive_Bayes = roc_gnb_test,
  Lasso_Regression = roc_lasso_test,
  XGBoost = roc_obj_xgboost,
  LightGBM = roc_obj_lightgbm_test,
  CatBoost = roc(test_label, pred_prob_cat_test)
)

roc_df_list <- lapply(names(rocobjs), function(method) {
  roc_obj <- rocobjs[[method]]
  data.frame(
    TPR = roc_obj$sensitivities,
    FPR = 1 - roc_obj$specificities,
    Method = method
  )
})

roc_df <- do.call(rbind, roc_df_list)
roc_df <- rbind(roc_df, dnn_roc_df)

methods_auc <- paste(
  c("Gaussian Naive Bayes", "Lasso Regression", "XGBoost", "LightGBM", "CatBoost", "Deep Neural Network"),
  "AUC =",
  round(c(auc(rocobjs$Naive_Bayes), auc(rocobjs$Lasso_Regression), auc(rocobjs$XGBoost), auc(rocobjs$LightGBM), auc(rocobjs$CatBoost), auc_value_dnn$AUC), 3)
)

ggplot_obj <- ggplot(roc_df, aes(x = FPR, y = TPR, color = Method)) +
  geom_line(size = 1, alpha = 0.7) +
  scale_color_discrete(labels = methods_auc) +
  labs(title = "ROC Curves for Different Models", x = "False Positive Rate", y = "True Positive Rate") +
  theme_minimal() +  
  theme(
    plot.title = element_text(size = 15),
    legend.title = element_blank(), 
    legend.text = element_text(size = 10)
  )

print(ggplot_obj)




ggplot_obj <- ggplot(roc_df, aes(x = FPR, y = TPR, color = Method)) + geom_line(size = 1.5, alpha = 0.8) + 
  scale_color_manual(values = c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00"), labels = methods_auc) + 
  labs(title = "ROC Curves for Different Models", x = "False Positive Rate", y = "True Positive Rate") + theme_minimal(base_size = 15) +
  theme( plot.title = element_text(hjust = 0.5, face = "bold"),  element_blank(), legend.text = element_text(size = 10), legend.position = "bottom"  ) 
print(ggplot_obj)

# plot for the train evaluation
library(ggplot2)

models <- c('GNB', 'GNB', 'GNB', 'GNB',
            'LASSO', 'LASSO', 'LASSO', 'LASSO',
            'XGBOOST', 'XGBOOST', 'XGBOOST', 'XGBOOST',
            'LIGHTGBM', 'LIGHTGBM', 'LIGHTGBM', 'LIGHTGBM',
            'CATBOOST', 'CATBOOST', 'CATBOOST', 'CATBOOST',
            'DNN', 'DNN', 'DNN', 'DNN')
metrics <- c('Accuracy', 'Precision', 'Recall', 'F1-score',
             'Accuracy', 'Precision', 'Recall', 'F1-score',
             'Accuracy', 'Precision', 'Recall', 'F1-score',
             'Accuracy', 'Precision', 'Recall', 'F1-score',
             'Accuracy', 'Precision', 'Recall', 'F1-score',
             'Accuracy', 'Precision', 'Recall', 'F1-score')
scores <- c(0.8629, 0.9656, 0.8888, 0.9256, 0.6218, 0.9729, 0.6230, 0.7596, 0.6483, 0.9810, 0.6457, 0.7788, 0.6743, 0.9840, 0.6712, 0.7981, 0.6161, 0.9771, 0.6140, 0.7542, 0.6467, 0.9724, 0.6499, 0.7791)


data <- data.frame(
  Model = factor(models, levels = c('GNB', 'LASSO', 'XGBOOST', 'LIGHTGBM', 'CATBOOST', 'DNN')),
  Metric = metrics,
  Score = scores
)

custom_colors <- c('Accuracy' = '#1f77b4', 'Precision' = '#ff7f0e', 'Recall' = '#2ca02c', 'F1-score' = '#d62728')

ggplot(data, aes(x = Model, y = Score, fill = Metric)) +
  geom_bar(position = position_dodge(width = 0.8), stat = "identity", width = 0.6) +
  geom_text(aes(label = round(Score, 2)), 
            position = position_dodge(width = 0.8), 
            vjust = -0.5, 
            size = 3) +
  scale_fill_manual(values = custom_colors) +
  labs(title = 'Training Performance', 
       x = 'Machine Learning Models', 
       y = 'Scores') +
  theme_minimal() +
  theme(legend.title = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1))

print(data)

# test evaluation
library(ggplot2)

# Correct Data for plotting
models <- c('GNB', 'GNB', 'GNB', 'GNB',
            'LASSO', 'LASSO', 'LASSO', 'LASSO',
            'XGBOOST', 'XGBOOST', 'XGBOOST', 'XGBOOST',
            'LIGHTGBM', 'LIGHTGBM', 'LIGHTGBM', 'LIGHTGBM',
            'CATBOOST', 'CATBOOST', 'CATBOOST', 'CATBOOST',
            'DNN', 'DNN', 'DNN', 'DNN')
metrics <- c('Accuracy', 'Precision', 'Recall', 'F1-score',
             'Accuracy', 'Precision', 'Recall', 'F1-score',
             'Accuracy', 'Precision', 'Recall', 'F1-score',
             'Accuracy', 'Precision', 'Recall', 'F1-score',
             'Accuracy', 'Precision', 'Recall', 'F1-score',
             'Accuracy', 'Precision', 'Recall', 'F1-score')
scores<- c(0.8630, 0.9654, 0.8889, 0.9256,  
           0.6199, 0.9721, 0.6214, 0.7581,  
           0.6403, 0.9770, 0.6398, 0.7732,  
           0.6637, 0.9773, 0.6646, 0.7912,  
           0.6127, 0.9755, 0.6112, 0.7515,  
           0.6604, 0.9700, 0.6663, 0.7900)   


data <- data.frame(
  Model = factor(models, levels = c('GNB', 'LASSO', 'XGBOOST', 'LIGHTGBM', 'CATBOOST', 'DNN')),
  Metric = metrics,
  Score = scores
)

custom_colors <- c('Accuracy' = '#1f77b4', 'Precision' = '#ff7f0e', 'Recall' = '#2ca02c', 'F1-score' = '#d62728')
