# ==============================================================================
# WEEK 3: STATISTICAL ANALYSIS & PREDICTIVE MODELING USING R
# ==============================================================================

# 1. Load Libraries & Dataset
install.packages(c("MASS", "tidyverse", "caret", "car", "lmtest", "corrplot"))
library(MASS)
library(tidyverse)
library(caret)
library(car)
library(lmtest)
library(corrplot)

data("Boston")
df <- Boston

# 2. Exploratory Data Analysis
str(df)
summary(df)
sum(is.na(df))

# Correlation Matrix
cor_matrix <- cor(df)
corrplot(cor_matrix, method = "color", type = "upper", tl.col = "black", tl.srt = 45)

# 3. Hypothesis Testing
cor.test(df$rm, df$medv, method = "pearson")
shapiro.test(df$medv[1:500])

# 4. Data Splitting & Cross-Validation Strategy
set.seed(123)
train_index <- createDataPartition(df$medv, p = 0.8, list = FALSE)
train_data  <- df[train_index, ]
test_data   <- df[-train_index, ]

# 5. Model Building with 10-Fold CV
train_control <- trainControl(method = "cv", number = 10)
model_cv <- train(
  medv ~ rm + lstat + ptratio + crim + tax,
  data = train_data,
  method = "lm",
  trControl = train_control
)

summary(model_cv$finalModel)

# 6. Diagnostics
vif(model_cv$finalModel)
par(mfrow = c(2, 2))
plot(model_cv$finalModel)
par(mfrow = c(1, 1))

# 7. Model Evaluation on Test Set
predictions <- predict(model_cv, newdata = test_data)
rmse_val <- RMSE(predictions, test_data$medv)
r2_val   <- R2(predictions, test_data$medv)

cat("Test Set RMSE:", rmse_val, "\n")
cat("Test Set R-Squared:", r2_val, "\n")
