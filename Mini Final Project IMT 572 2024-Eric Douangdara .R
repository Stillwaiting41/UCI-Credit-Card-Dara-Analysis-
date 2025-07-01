# Install Packages
install.packages("randomForest")
install.packages("mlbench")
install.packages("naivebayes")
install.packages("Hmisc")
install.packages("e1071")
install.packages("party")
install.packages("mfx")
install.packages("tidyr")
install.packages("broom")
install.packages("mfx")
install.packages("reshape2")
# Load Libraries
library(tidyr)
library(randomForest)
library(mlbench)
library(naivebayes)
library(Hmisc)
library(dplyr)
library(ggplot2)
library(caret)
library(kernlab)
library(broom)
library(mfx)
library(reshape2)

# Load the dataset
getwd()
setwd("c:/Users/douan/Downloads/IMT572")
data <- read.csv("default of credit card clients.csv")

# Check structure and summary of the dataset
str(data)
summary(data)
colnames(data)

# Rename the target variable for clarity
colnames(data)[colnames(data) == "default.payment.next.month"] <- "default_payment"

# Convert categorical variables to factors
data$SEX <- factor(data$SEX, levels = c(1, 2), labels = c("Male", "Female"))
data$EDUCATION <- factor(data$EDUCATION)
data$MARRIAGE <- factor(data$MARRIAGE)
data$default_payment <- factor(data$default_payment, levels = c(0, 1), labels = c("No", "Yes"))

# Normalize numerical variables
normalize <- function(x) (x - min(x)) / (max(x) - min(x))
data[, c("LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1")] <- 
  lapply(data[, c("LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1")], normalize)

normalize <- data[, c("LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1")]
normalize <- pivot_longer(as.data.frame(normalize), cols = everything(), names_to = "Variable", values_to = "Value")

# Calculate the mean for all numeric columns
mean_values <- sapply(data, function(x) if (is.numeric(x)) mean(x, na.rm = TRUE))

# View the means
print(mean_values)

# Select numeric variables from the dataset
numeric_data <- data[, sapply(data, is.numeric)]

# Logistic Regression Model
logit_model = glm(default_payment ~ LIMIT_BAL + AGE + SEX + EDUCATION + MARRIAGE + BILL_AMT1 + PAY_AMT1,
                  family="binomial",data = data)
summary(logit_model)

# interpret the marginal effects table.
logitmfx(default_payment~PAY_0+ PAY_2+PAY_3+EDUCATION+LIMIT_BAL+BILL_AMT1, data = data)

# Probit Regression Model
probit_model <- glm(default_payment ~ LIMIT_BAL + AGE + SEX + EDUCATION + MARRIAGE + BILL_AMT1 + PAY_AMT1,
                    family = binomial(link = "probit"), data = data)

# Summary of the probit regression model
summary(probit_model)

# Compare models using AIC
AIC(logit_model, probit_model)

print(AIC)
# Split the data into training and testing sets
set.seed(123)
train_index <- createDataPartition(data$default_payment, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Set up control for cross-validation
control <- trainControl(method = "cv", number = 5)

# K-Nearest Neighbors (KNN)
model_knn <- train(default_payment ~ LIMIT_BAL + AGE + SEX + EDUCATION + MARRIAGE + BILL_AMT1 + PAY_AMT1,
                   data = train_data, method = "knn", trControl = control, tuneLength = 10)

# Support Vector Machine (Linear)
model_svm <- train(default_payment ~ LIMIT_BAL + AGE + SEX + EDUCATION + MARRIAGE + BILL_AMT1 + PAY_AMT1,
                   data = train_data, method = "svmLinear", trControl = control, tuneLength = 10)

# KNN Evaluation
knn_predictions <- predict(model_knn, newdata = test_data)
confusionMatrix(knn_predictions, test_data$default_payment)

# SVM Evaluation
svm_predictions <- predict(model_svm, newdata = test_data)
confusionMatrix(svm_predictions, test_data$default_payment)

# Compare accuracy
knn_acc <- postResample(knn_predictions, test_data$default_payment)[1]
svm_acc <- postResample(svm_predictions, test_data$default_payment)[1]

cat("KNN Accuracy:", knn_acc, "\n")
cat("SVM Accuracy:", svm_acc, "\n")
print(cat)
print(svm_acc)

data$default_payment = as.factor(data$default_payment)

train_Control =trainControl(method="cv",number= 3)
knn_caret = train(default_payment~PAY_0+ PAY_2+PAY_3+EDUCATION+LIMIT_BAL+BILL_AMT1, 
                  data = data, 
                  method = "knn", trControl = train_Control,
                  tuneLength = 3)
plot(knn_caret)

# Visualizations

# Create histograms
ggplot(normalize, aes(x = Value)) +
  geom_histogram(bins = 30, fill = "blue", color = "black") +
  facet_wrap(~Variable, scales = "free") +
  theme_minimal() +
  ggtitle("Histograms of Normalized Variables") +
  xlab("Normalized Value") +
  ylab("Frequency")

# Boxplot of Age by Default Status
ggplot(data, aes(x=factor(default_payment), y=AGE)) +
  geom_boxplot(fill=c("lightblue", "lightcoral")) +
  theme_minimal() +
  labs(title="Age Distribution by Default Status", x="Default Payment (0 = No, 1 = Yes)", y="Age")

# Boxplot of Credit Limit by Default Status
ggplot(data, aes(x=factor(default_payment), y=LIMIT_BAL)) +
  geom_boxplot(fill=c("lightblue", "lightcoral")) +
  theme_minimal() +
  labs(title="Credit Limit Distribution by Default Status", x="Default Payment (0 = No, 1 = Yes)", y="Credit Limit")

# Boxplot for Credit Amount by Education Level
ggplot(data, aes(x = EDUCATION, y = LIMIT_BAL)) + 
  geom_boxplot() + 
  theme_minimal() + 
  ggtitle("Boxplot of Credit Amount by Education Level")

# Bar Plot for SEX Distribution
ggplot(data, aes(x = SEX)) + 
  geom_bar(fill = "lightblue", color = "lightcoral") + 
  theme_minimal() + 
  ggtitle("SEX Distribution")

# Scatter plot of Logsitic progression model
data$predicted_probability <- predict(logit_model,type = "response")
ggplot(data, aes(x=LIMIT_BAL, y= default_payment)) +
geom_point(alpha = 0.3 ,color="lightblue",aes(size = 0.5),shape= 16) +
geom_smooth(aes (y = predicted_probability), method = "glm", method.args= list (family = "binomial"), color = "red", se= FALSE)+
labs(title = "Logistic Regression: Default payment vs Credit Limit",
x= "Credit Limit(LIMIT_BAL",
y= "Default Payment Probability")+
theme_minimal()

#Scatter plot of Probit Regression Model
data$predicted_probability_probit <- predict(probit_model,type = "response")
ggplot(data, aes(x=LIMIT_BAL, y= default_payment)) +
  geom_point(alpha = 0.3 ,color="lightblue",aes(size = 0.5),shape= 16) +
  geom_line(aes (y = predicted_probability_probit), color = "red", se= FALSE)+
  labs(title = "Probit Regression: Default payment vs Credit Limit",
       x= "Credit Limit(LIMIT_BAL",
       y= "Default Payment Probability")+
  theme_minimal()

# Create cross-validation graph
data$default_payment = as.factor(data$default_payment)

train_Control =trainControl(method="cv",number= 3)
knn_caret = train(default_payment~PAY_0+ PAY_2+PAY_3+EDUCATION+LIMIT_BAL+BILL_AMT1, 
                  data = data, 
                  method = "knn", trControl = train_Control,
                  tuneLength = 3)
plot(knn_caret)