# ML code- Runze Wang

summary(DHS_ARI_runze)


library(labelled)
library(dplyr)
val_labels(`DHS_ARI_runze`$v190)
table(DHS_ARI_runze$ch_allvac_either)
table(DATA2$year)

df <- subset(DHS_ARI_runze, select = -c(P_ID, v000, Surv_year, wt, ADM1_NAME, mat_age_cat, year, v024, ch_allvac_either, ch_allvac_moth, ch_allvac_card))

df$URBAN_RURA <- ifelse(df$URBAN_RURA == "R", 1, ifelse(df$URBAN_RURA == "U", 0, NA))
table(df$URBAN_RURA)


#b4 1,2 into 0,1
table(DHS_ARI_runze$b4)
typeof(df$b4)
df$b4 <- df$b4 - 1 
table(df$b4)
typeof(df$b4)

table(DHS_ARI_runze$v190)
df$v190 <- df$v190 - 1 
table(df$v190)


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
df2_new1<- subset(df2, select = -c(State))

#rename 
df2_new<- df2_new1 %>% rename(
  birth_order = bord,
  gender = b4,
  age_of_child_in_years = b8,
  mother_education = rc_edu,
  mother_employment = rc_empl,
  media_use = rc_media_allthree,
  health_insurance = rc_hins_any,
  mother_use_of_tobacco_products = rc_tobc_any,
  age_of_mother_at_birth = mat_age,
  household_wealth = v190,
  Angola= StateAO,
  Burkina_Faso = StateBF,
  Benin = StateBJ,
  Burundi = StateBU,
  Congo_Democratic_Republic = StateCD,
  Cameroon = StateCM,
  Gabon = StateGA,
  Ghana = StateGH,
  Guinea = StateGN,
  Kenya =  StateKE,
  Comoros = StateKM,
  Liberia = StateLB,
  Lesotho = StateLS,
  Madagascar = StateMD,
  Mali = StateML,
  Malawi = StateMW,
  Mozambique = StateMZ,
  Nigeria = StateNG,
  Niger = StateNI,
  Namibia = StateNM,
  Rwanda = StateRW,
  Sierra_Leone = StateSL,
  Senegal = StateSN,
  Swaziland = StateSZ,
  Chad = StateTD,         
  Togo = StateTG,
  Tanzania = StateTZ,
  Uganda = StateUG,
  Zambia = StateZM,
  Zimbabwe = StateZW,
  urban_and_rural_residence = URBAN_RURA,
  Latitudinal = LATNUM,
  Longitudinal = LONGNUM,
  ARI = ch_ari)

print(df2_new1)
df2_new$mother_education <- as.numeric(as.character(df2_new$mother_education))
df2_new$mother_employment <- as.numeric(as.character(df2_new$mother_employment))
df2_new$household_wealth <- as.numeric(as.character(df2_new$household_wealth))
df2_new$media_use <- as.numeric(as.character(df2_new$media_use))
df2_new$mother_use_of_tobacco_products <- as.numeric(as.character(df2_new$mother_use_of_tobacco_products))
str(df2_new)
unique(as.character(df2_new$mother_employment))


# not use
df2_cor<- df2_new %>% rename(
  birth_order = bord,
  gender = b4,
  age_of_child_in_years = b8,
  mother_education = rc_edu,
  mother_employment = rc_empl,
  media_use = rc_media_allthree,
  health_insurance = rc_hins_any,
  mother_use_of_tobacco_products = rc_tobc_any,
  age_of_mother_at_birth = mat_age,
  household_wealth = v190,
  ARI = ch_ari)



#correlation
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
df2_new$ARI <- as.factor(df2_new$ARI)
information_gain_scores <- information_gain(ARI ~ ., df2_new)
importance_scores <- information_gain_scores$importance
importance_scores
summary(importance_scores)
hist(importance_scores, main="Distribution of Information Gain Scores", xlab="Information Gain", breaks=30)
threshold <- quantile(importance_scores, probs = 0.50)
print(threshold)
significant_features <- information_gain_scores %>%
  filter(importance > threshold) %>%
  pull(attributes)
print(significant_features)
write.csv(information_gain_scores, "importance_scores_0.5.csv", row.names = FALSE)
write.csv(significant_features, "significant_features_0.5.csv", row.names = FALSE)
df2_new$ARI <- as.numeric(as.character(df2_new$ARI))
class(df2_new$ARI)

#depend on the result of information gain

df2_new <- subset(df2_new, select = -c(
  birth_order, gender, media_use, mother_employment, health_insurance, mother_use_of_tobacco_products, 
  age_of_mother_at_birth, IU_Preci_z, PU_Preci_z, IU_SO, Angola, Benin, 
  Congo_Democratic_Republic, Cameroon, Gabon, Ghana, Guinea, Kenya, Comoros, Liberia, 
  Lesotho, Madagascar, Malawi, Mozambique, Niger, Namibia, Rwanda, Sierra_Leone, Senegal, 
  Swaziland, Chad, Togo, Tanzania, Zambia, Zimbabwe
))

colnames(df2_new)



# Load the necessary library
library(gtsummary)

# Assuming your dataset is called `dataset`
# Create a summary table
table_summary <- df2_new %>%
  tbl_summary(
    by = ARI, # ARI is the outcome variable
    statistic = list(all_continuous() ~ "{mean} ({sd})", # For continuous variables
                     all_categorical() ~ "{n} ({p}%)"),  # For categorical variables
    label = list(
      age_of_child_in_years ~ "Age of Child (Years)",
      mother_education ~ "Mother's Education",
      household_wealth ~ "Household Wealth",
      IU_Tmp_z ~ "IU Temperature (z-score)",
      PU_Tmx_z ~ "PU Max Temperature (z-score)",
      IU_BC ~ "IU Black Carbon",
      PU_BC ~ "PU Black Carbon",
      urban_and_rural_residence ~ "Residence Type"
    ),
    missing = "no" # Choose how to handle missing data
  ) %>%
  add_p() # Add p-values for comparison between ARI groups

# Print the summary table
table_summary





library(caret)
set.seed(123)
train_indices<- createDataPartition(df2_new $ARI, p = .7, list = FALSE, times = 1)
str(train_indices)
train_indices <- as.vector(train_indices[, 1])
train_data <- df2_new[train_indices, ]
test_data <- df2_new[-train_indices, ]


#_standardized (for none tree)
train_data_standardized <-train_data 
train_data_standardized [, 1] <- scale(train_data [, 1], center = TRUE, scale = TRUE)
train_data_standardized [, 3:29] <- scale(train_data [, 3:29], center = TRUE, scale = TRUE)
head(train_data_standardized )

test_data_standardized  <-test_data 
test_data_standardized  [, 1] <- scale(test_data  [, 1], center = TRUE, scale = TRUE)
test_data_standardized  [, 3:29] <- scale(test_data  [, 3:29], center = TRUE, scale = TRUE)

head(test_data_standardized )
# data imblance 
library(ggplot2)
ggplot(data = train_data_standardized, aes(x = ch_ari, fill = as.factor(ch_ari))) + 
  geom_bar(stat = "count", width = 0.7) +  
  ggtitle("Number of samples in each class", subtitle = "Original dataset") +
  xlab("") +
  ylab("Samples") +
  scale_y_continuous(expand = c(0, 0)) +
  scale_x_discrete(expand = c(0, 0)) +
  scale_fill_manual(values = c("#FF686B", "#4BC0C8")) + 
  theme(
    legend.position = "none",  
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.ticks = element_blank(),
    axis.text.x = element_text(angle = 0, hjust = 0.5, vjust = 0.5)
  )
