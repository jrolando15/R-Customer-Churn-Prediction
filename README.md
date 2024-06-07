# R-Customer-Churn-Prediction

# Project Description
This project aims to predict customer churn using various machine learning models. The dataset used contains customer information and their churn status. The project involves data preprocessing, rebalancing the dataset, and building multiple classification models to predict churn.

# Table of Content
- [Installation](#installation)
- [Usage](#usage)
- [Data Processing](#data-processing)
- [Financial Analysis](#financial-analysis)
- [Sentiment Analysis](#sentiment-analysis)
- [Results](#results)
- [License](#license)

# Installation
To run this project, you need to have R installed on your system. Additionally, you need to install the following R packages:
```R
install.packages(c("dplyr", "rpart", "rpart.plot", "C50", "randomForest", "party"))
```

# Usage
1. Clone the repository to your local machine.
2. Load the dataset and run the preprocessing steps.
3. Train the models and evaluate their performance.

#  Data Preprocessing
The dataset is loaded and preprocessed as follows:

1. load the dataset
```R
churn <- read.csv("~path to the csav file")
```
2. Remove the 'phone' column
```R
library(dplyr)
churn_liu <- select(churn, -Phone)
```

3. Add an index column:
```R
N <- dim(churn_liu)[1]
churn_liu$index <- c(1:N)
```

4. Split the data into training and testing sets:
```R
set.seed(7)
n <- dim(churn_liu)[1]
train_ind <- runif(n) < .67
churn_train <- churn_liu[train_ind,]
churn_test <- churn_liu[!train_ind,]
```

5. Rebalance the trainig and testing datasets:
```R
# Rebalance training data
to.resample <- which(churn_train$Churn == "True")
our.resample <- sample(x = to.resample, size = 175, replace = TRUE)
our.resample <- churn_train[our.resample, ]
train_churn_rebal <- rbind(churn_train, our.resample)

# Rebalance testing data
to.resample_1 <- which(churn_test$Churn == "True")
our.resample_1 <- sample(x = to.resample_1, size = 63, replace = TRUE)
our.resample_1 <- churn_test[our.resample_1, ]
test_churn_rebal <- rbind(churn_test, our.resample_1)
```

6. Convert neccesary columns to factors
```R
churn_train$Churn <- as.factor(churn_train$Churn)
churn_test$Churn <- as.factor(churn_test$Churn)
churn_train$State <- as.factor(churn_train$State)
churn_test$State <- as.factor(churn_test$State)
churn_train$VMail.Plan <- as.factor(churn_train$VMail.Plan)
churn_test$VMail.Plan <- as.factor(churn_test$VMail.Plan)
```

# Model Training 
Several models are trained to predict customer churn:

1. Cart Models
```R
library(rpart)
library(rpart.plot)

# Model 1
cart_model_1 <- rpart(Churn ~ Day.Mins + CustServ.Calls, data = churn_train, method = "class")
rpart.plot(cart_model_1)

# Model 2
cart_model_2 <- rpart(Churn ~ VMail.Plan + Day.Charge, data = churn_train, method = "class")
rpart.plot(cart_model_2)

# Model 3
cart_model_3 <- rpart(Churn ~ State + Day.Mins, data = churn_train, method = "class")
rpart.plot(cart_model_3)
```
2. C5.0 Models
```R
library(C50)

# Model 1
C5 <- C5.0(formula = Churn ~ Day.Mins + CustServ.Calls, data = churn_train, control = C5.0Control(minCases = 100))
plot(C5)

# Model 2
C7 <- C5.0(formula = Churn ~ VMail.Plan + Day.Charge, data = churn_train, control = C5.0Control(minCases = 100))
plot(C7)

# Model 3
C9 <- C5.0(formula = Churn ~ State + Day.Mins, data = churn_train, control = C5.0Control(minCases = 100))
plot(C9)
```

3. Random Forest Models:
```R
library(randomForest)

# Model 1
rf05 <- randomForest(formula = Churn ~ State + Day.Mins, data = churn_train, ntree = 100, type = "classification")

# Model 2
rf06 <- randomForest(formula = Churn ~ State + Day.Mins, data = churn_test, ntree = 100, type = "classification")
```

4. Conditional Inference Trees:
```R
library(party)

# Model 1
X_5 <- ctree(Churn ~ State + Day.Mins, data = churn_train)
plot(X_5, type = 'simple')

# Model 2
X_6 <- ctree(Churn ~ State + Day.Mins, data = churn_test)
plot(X_6, type = "simple")
```

# Model Evaluation

1. Cart model Evaluation
```R
cart_model_test_1 <- rpart(Churn ~ Day.Mins + CustServ.Calls, data = churn_test, method = "class")
rpart.plot(cart_model_test_1)

cart_model_test_2 <- rpart(Churn ~ VMail.Plan + Day.Charge, data = churn_test, method = "class")
rpart.plot(cart_model_test_2)

cart_model_test_3 <- rpart(Churn ~ State + Day.Mins, data = churn_test, method = "class")
rpart.plot(cart_model_test_3)
```

2. C5.0 Model evaluation
```R
C6 <- C5.0(formula = Churn ~ Day.Mins + CustServ.Calls, data = churn_test, control = C5.0Control(minCases = 100))
plot(C6)

C8 <- C5.0(formula = Churn ~ VMail.Plan + Day.Charge, data = churn_test, control = C5.0Control(minCases = 100))
plot(C8)

C10 <- C5.0(formula = Churn ~ State + Day.Mins, data = churn_test, control = C5.0Control(minCases = 100))
plot(C10)
```

3. Random Forest Evaluation
```R
summary(rf05)
summary(rf06)
```

4. Condtional Tree
```R
summary(X_5)
summary(X_6)
```

# License
This README template includes all the pertinent information about your project, such as installation instructions, usage, project structure, data processing, model training, model evaluation, and details about the web application. It also includes sections for contributing and licensing, which are important for open-source projects.
