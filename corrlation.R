library(labelled)
library(dplyr)
val_labels(`DATA2`$v190)
table(DATA2$v190)
table(DATA2$year)

df <- subset(DATA2, select = -c(ID, ADM1NAME,caseid, State2, CLUSTID, Surv_year,wt,CC_CLUSTID,mat_age_cat,year))

#b4 1,2 into 0,1
table(DATA2$b4)
typeof(df$b4)
df$b4 <- df$b4 - 1 
table(df$b4)
typeof(df$b4)

# URBAN_RURA into 0,1 R=1 U=0
df$URBAN_RURA <- ifelse(df$URBAN_RURA == "R", 1, ifelse(df$URBAN_RURA == "U", 0, NA))
table(df$URBAN_RURA)
# remove missing value 
sum(is.na(df))
df2 <- na.omit(df)
sum(is.na(df2))

#change the variable into  categories- prepare for the one hot encoding
# Convert each variable to factor individually
df2$State <- as.factor(df2$State)

one_hot_encoded <- model.matrix(~ State    - 1, data = df2)
head(one_hot_encoded)
df2<- cbind(df2, one_hot_encoded)
df2_new <- subset(df2, select = -c(State))

#rename 
df2_new<- df2_new %>% rename(
  birth_order = bord,
  gender = b4,
  age_of_child_in_years = b8,
  media_use = rc_media_allthree,
  mother_use_of_tobacco_products = rc_tobc_any,
  age_of_mother_at_birth = mat_age,
  urban_and_rural_residence = URBAN_RURA,
  mother_education = rc_edu,
  mother_employment = rc_empl,
  household_wealth = v190
)
print(df2_new)
str(df2_new)
df2_new$mother_education <- as.numeric(as.character(df2_new$mother_education))
df2_new$mother_employment <- as.numeric(as.character(df2_new$mother_employment))
df2_new$household_wealth <- as.numeric(as.character(df2_new$household_wealth))
variable_names <- colnames(df2_new)
print(variable_names)

library(ggcorrplot)
cor_matrix2 <- data.frame(cor(df2_new))
cor_matrix2
corr_plot <- ggcorrplot(cor_matrix2, lab = FALSE) + 
  theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 6, margin = margin(t = 10)),
    axis.text.y = element_text(size = 6, margin = margin(r = 10))
  )
print(corr_plot)
library(pheatmap)


# Information Gain
library(FSelectorRcpp)
library(dplyr)
df2_new$ch_ari <- as.factor(df2_new$ch_ari)
information_gain_scores <- information_gain(ch_ari ~ ., df2_new)
importance_scores <- information_gain_scores$importance
importance_scores
summary(importance_scores)
hist(importance_scores, main="Distribution of Information Gain Scores", xlab="Information Gain", breaks=30)
threshold <- quantile(importance_scores, probs = 0.25)
print(threshold)
significant_features <- information_gain_scores %>%
  filter(importance > threshold) %>%
  pull(attributes)
print(significant_features)
write.csv(information_gain_scores, "importance_scores.csv", row.names = FALSE)
write.csv(significant_features, "significant_features.csv", row.names = FALSE)
df2_new$ch_ari <- as.numeric(as.character(df2_new$ch_ari))
class(df2_new$ch_ari)
df2_new <- subset(df2_new, select = -c(birth_order, gender, media_use, mother_use_of_tobacco_products, age_of_mother_at_birth, StateAO, StateCD, StateKM, StateLS, StateNI, StateNM, StateRW, StateSL, StateTG, StateTZ, StateZM, StateZW))



library(caret)
set.seed(123)
train_indices<- createDataPartition(df2_new $ch_ari, p = .7, list = FALSE, times = 1)
str(train_indices)
train_indices <- as.vector(train_indices[, 1])
train_data <- df2_new[train_indices, ]
test_data <- df2_new[-train_indices, ]


#_standardized
train_data_standardized <-train_data 
train_data_standardized [, 1] <- scale(train_data [, 1], center = TRUE, scale = TRUE)
train_data_standardized [, 4:34] <- scale(train_data [, 4:5], center = TRUE, scale = TRUE)
head(train_data_standardized )

test_data_standardized  <-test_data 
test_data_standardized  [, 1] <- scale(test_data  [, 1], center = TRUE, scale = TRUE)
test_data_standardized  [, 4:34] <- scale(test_data  [, 4:34], center = TRUE, scale = TRUE)

head(test_data_standardized )

library(xgboost)
library(lime)

train_data_xgboost <- as.matrix(train_data_standardized[, -which(names(train_data_standardized) == "ch_ari")])
train_label <- train_data_standardized$ch_ari
test_data_xgboost <- as.matrix(test_data_standardized[, -which(names(test_data_standardized) == "ch_ari")])
test_label <- test_data_standardized$ch_ari

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
pred_class <- ifelse(pred_probs > 0.6, 1, 0)
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
