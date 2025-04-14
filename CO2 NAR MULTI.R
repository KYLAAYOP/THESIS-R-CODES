#' ---
#' title: " CO2 NAR MULTI "
#' author: "Kyla M. Ayop"
#' date: "`r Sys.Date()`"
#' output: html_document
#' ---
#' 
#' Load Required Packages
## ----warning=FALSE, message=FALSE-----------------------------
#Load Required Packages
library(kernlab)    # Gaussian Process Regression
library(ggplot2)    # Visualization
library(dplyr)      # Data manipulation
library(tidyr)      # Data wrangling
library(stats)


#Load and Explore Dataset
data <- read.csv("co2_mm_mlo.csv", header = TRUE)
head(data)
tail(data)

#Pre-processing of Data
# Convert Year and Month to Date format
data$Time <- as.Date(paste(data$Year, data$Month, "01", sep="-"))

#Centering the Data
#To center the data (subtract the mean):Compute the mean of CO2 concentration
ts_dt <- ts(data$Average,frequency = 12, start = c(min(data$Year), 3))
mean_co2 <- mean(ts_dt, na.rm = TRUE)
data$centered_CO2 <- ts_dt - mean_co2

#Standardizing the Data
#To standardize the data (subtract the mean and divide by the standard deviation):Compute the standard deviation
sd_co2 <- sd(data$Average, na.rm = TRUE)
data$standardized_CO2 <- (data$Average - mean_co2) / sd_co2
head(data)

ts_data <- ts(data$Average, frequency = 12, start = c(min(data$Year), 3))

# Plot time series
ggplot(data, aes(x = Time, y = centered_CO2)) +
  geom_line() +
  labs(title = "Co2 Averages Over Time", x = "Time", y = "centered_CO2") +
  theme_minimal()

# Plot time series
ggplot(data, aes(x = Time, y = standardized_CO2)) +
  geom_line() +
  labs(title = "Co2 Averages Over Time", x = "Time", y = "standardized_CO2") +
  theme_minimal()

# Plot time series
ggplot(data, aes(x = Time, y = Average)) +
  geom_line() +
  labs(title = "Co2 Averages Over Time", x = "Time", y = "Average") +
  theme_minimal()

#' 
#' Evaluate Different Lag Orders
#' PACF Plot and CV for GP-NAR
## -------------------------------------------------------------
# Function to plot PACF and determine initial lag
get_pacf_lag <- function(ts_data) {
  pacf(ts_data, main = "PACF Plot for Time Series")
}
# Generate PACF plot and determine the initial lag
get_pacf_lag(ts_data)  # Analyze the PACF plot and visually pick a starting lag

#' 
#' Lag Evaluation
## -------------------------------------------------------------
library(kernlab)
library(caret)

# PACF-based range of lag orders to test
min_lag <- 2
max_lag <- 20
h <- 1  # One-step ahead

# Store results
lag_cv_results <- data.frame()

set.seed(123)  # For reproducibility

for (lag_order in min_lag:max_lag) {
  # Embed the time series
  embedded_matrix <- embed(ts_data, lag_order + h)
  X <- embedded_matrix[, -(1:h)]  # Lag features
  Y <- embedded_matrix[, 1]       # Target
  colnames(X) <- paste0("Lag_", 1:ncol(X))

  # Use 5-fold cross-validation on the training set
  split_index <- floor(0.8 * nrow(X))
  X_train <- X[1:split_index, ]
  Y_train <- Y[1:split_index]

  k_folds <- 5
  folds <- createFolds(Y_train, k = k_folds, list = TRUE)

  rmse_fold <- c()

  for (i in 1:k_folds) {
    val_idx <- folds[[i]]
    X_cv_train <- X_train[-val_idx, , drop = FALSE]
    Y_cv_train <- Y_train[-val_idx]
    X_cv_val <- X_train[val_idx, , drop = FALSE]
    Y_cv_val <- Y_train[val_idx]

    if (nrow(X_cv_train) > ncol(X_cv_train)) {
      fit <- tryCatch({
        gausspr(X_cv_train, Y_cv_train, kernel = rbfdot(sigma = 1))
      }, error = function(e) {
        cat("❌ Fit error for lag =", lag_order, ": ", e$message, "\n")
        return(NULL)
      })

      if (!is.null(fit)) {
        y_pred <- predict(fit, X_cv_val)
        rmse <- sqrt(mean((Y_cv_val - y_pred)^2))
        rmse_fold <- c(rmse_fold, rmse)
      } else {
        rmse_fold <- c(rmse_fold, Inf)
      }
    } else {
      rmse_fold <- c(rmse_fold, Inf)
    }
  }

  avg_rmse <- mean(rmse_fold)
  lag_cv_results <- rbind(
    lag_cv_results,
    data.frame(Lag = lag_order, RMSE = avg_rmse)
  )

  cat("✅ Lag =", lag_order, "| Mean CV RMSE =", round(avg_rmse, 4), "\n")
}
best_lag_row <- lag_cv_results[which.min(lag_cv_results$RMSE), ]
best_lag_order <- best_lag_row$Lag
cat("\n🏆 Best Lag =", best_lag_order, "with RMSE =", round(best_lag_row$RMSE, 4), "\n")


#' 
#' Time Series Embedding
## -------------------------------------------------------------
# Load necessary library
library(stats)

# Define parameters
p <- best_lag_order  # Window length (lag)
h <- 1  # Steps ahead for forecasting

# Embedding procedure
embedded_matrix <- embed(ts_data, p + h)

# Split into X (inputs) and Y (output target)
X <- embedded_matrix[, -(1:h)]  # Input matrix
Y <- embedded_matrix[, 1]       # Output vector
colnames(X) <- paste0("Lag_", 1:ncol(X))

# Split data into training and testing sets
# Calculate the split index for 80% training and 20% testing
split_index <- floor(0.8 * nrow(X))  # 80% for training

# Split data into training and testing sets
X_train <- X[1:split_index, ]  # First 80% as training data
Y_train <- Y[1:split_index]

X_test <- X[(split_index + 1):nrow(X), , drop = FALSE]  # Remaining 20% as testing data
Y_test <- Y[(split_index + 1):length(Y)]

# Display results
print("Training Inputs (X_train):")
head(X_train)
print("Training Outputs (Y_train):")
head(Y_train)
print("Testing Inputs (X_test):")
head(X_test)
print("Testing Outputs (Y_test):")
head(Y_test)

#' 
#' Kernel Selection
## -------------------------------------------------------------
x <- as.numeric(data$Time)
y <- data$Average

#' 
## -------------------------------------------------------------
set.seed(123)

# Define Isotropic RBF Kernel
rbf_isotropic <- function(x1, x2, lambda , sigma ) {
  # Compute squared Euclidean distance between two points
  distance_squared <- sum((x1 - x2)^2)
  # Kernel formula for isotropic RBF
  scaling_factor <- 1 / lambda^2
  return(sigma^2 * exp(-0.5 * scaling_factor * distance_squared))
}
class(rbf_isotropic) <- "kernel"  # Set as kernel class

gp_model_rbf_isotropic <- function(params) {
  lambda <- params[1]
  sigma <- params[2]
  # Kernel matrix using custom isotropic RBF kernel
  n <- length(y)  # Get the number of data points
  # Kernel matrix using isotropic RBF kernel
  K <- matrix(0, nrow = n, ncol = n)
  for (i in 1:n) {
    for (j in 1:n) {
      K[i, j] <- rbf_isotropic(x[i], x[j], lambda, sigma)
    }
  }
  K <- K + diag(1e-6, n)  # Numerical stability
  
  # Compute log marginal likelihood
  L <- chol(K)
  alpha <- solve(t(L), solve(L, y))
  log_likelihood <- -0.5 * t(y) %*% alpha - sum(log(diag(L))) - (n / 2) * log(2 * pi)
  return(-log_likelihood)  # Negative for optimization
}

opt_result_rbf_iso <- optim(
  par = c(lambda = 1, sigma = 1),
  fn = gp_model_rbf_isotropic,
  method = "L-BFGS-B",
  lower = c(0.1, 0.1),
  upper = c(10, 10)
)
print(opt_result_rbf_iso$par)
# Print Optimal Parameters
cat("Optimal Parameters:\n")
cat("Length Scales:", opt_result_rbf_iso$par[1], "\n")
cat("Sigma:", opt_result_rbf_iso$par[2], "\n")

# Extract Optimal Parameters
lambda_opt <- opt_result_rbf_iso$par[1]  # Length scale (lambda)
sigma_opt <- opt_result_rbf_iso$par[2]   # Sigma (amplitude)

# Define Isotropic RBF Kernel with Optimal Parameters
rbf_iso_with_opt <- function(x1, x2) {
  # Use the optimal parameters directly in the kernel computation
  distance_squared <- sum((x1 - x2)^2)
  scaling_factor <- 1 / lambda_opt^2
  return(sigma_opt^2 * exp(-0.5 * scaling_factor * distance_squared))
}
class(rbf_iso_with_opt) <- "kernel"  # Set as kernel class


#' 
## -------------------------------------------------------------
set.seed(123)

# Define Isotropic Rational Quadratic (RQ) Kernel
rq_isotropic <- function(x1, x2, alpha_p, lambda , sigma ) {
  # Compute squared Euclidean distance between two points
  distance_squared <- sum((x1 - x2)^2)
  # Kernel formula for isotropic RQ
  scaling_factor <- 1 / lambda ^2
  factor <- 1 + (scaling_factor * distance_squared) / (2 * alpha_p)
  return(sigma ^2 * factor^(-alpha_p))
}
class(rq_isotropic) <- "kernel"  # Set as kernel class

gp_model_rq_isotropic <- function(params) {
  alpha_p <- params[1]
  lambda <- params[2]
  sigma <- params[3]
  # Kernel matrix using custom isotropic RBF kernel
  n <- length(y)  # Get the number of data points
  # Kernel matrix using isotropic RQ kernel
  K <- matrix(0, nrow = n, ncol = n)
  for (i in 1:n) {
    for (j in 1:n) {
      K[i, j] <- rq_isotropic(x[i], x[j], alpha_p, lambda, sigma)
    }
  }
  K <- K + diag(1e-6, n)  # Numerical stability
  
  # Compute log marginal likelihood
  L <- chol(K)
  alpha <- solve(t(L), solve(L, y))
  log_likelihood <- -0.5 * t(y) %*% alpha - sum(log(diag(L))) - (n / 2) * log(2 * pi)
  return(-log_likelihood)  # Negative for optimization
}

opt_result_rq_iso <- optim(
  par = c(alpha_p = 1, lambda = 1, sigma = 1),
  fn = gp_model_rq_isotropic,
  method = "L-BFGS-B",
  lower = c(0.1, 0.1, 0.1),
  upper = c(10, 10, 10)
)

print(opt_result_rq_iso$par)
# Print Optimal Parameters
cat("Optimal Parameters:\n")
cat("Alpha:", opt_result_rq_iso$par[1] , "\n")
cat("Length Scales:", opt_result_rq_iso$par[2], "\n")
cat("Sigma:", opt_result_rq_iso$par[3]    , "\n")

# Optimal Parameters from GP Optimization
alpha_opt <- opt_result_rq_iso$par[1]        # Alpha
lambda_opt <- opt_result_rq_iso$par[2]  # Vector of 3 Length Scales
sigma_opt <- opt_result_rq_iso$par[3]        # Sigma

# ARD Rational Quadratic (RQ) Kernel with Optimal Parameters
rq_iso_with_opt <- function(x1, x2) {
  # Use the optimal parameters in the kernel computation
   distance_squared <- sum((x1 - x2)^2)
   M <- 1 / lambda_opt^2  # Isotropic case
   factor <- 1 + (M * distance_squared) / (2 * alpha_opt)
   return(sigma_opt^2 * factor^(-alpha_opt))
 }

class(rq_iso_with_opt) <- "kernel"  # Set as kernel class

#' 
## -------------------------------------------------------------
x <- X
y <- Y

#' 
## ----warning=FALSE--------------------------------------------
set.seed(123)

# Define ARD RBF Kernel
rbf_ard <- function(x1, x2, length_scales, sigma) {
  # Compute squared Euclidean distance with ARD (Automatic Relevance Determination)
  distance_squared <- sum(((x1 - x2) / length_scales)^2)
  return(sigma^2 * exp(-0.5 * distance_squared))
}
class(rbf_ard) <- "kernel"  # Set as kernel class

# GP Model with ARD RBF Kernel
gp_model_rbf_ard <- function(params) {
  length_scales <- params[1:ncol(x)]  # Extract the vector of 3 length scales
  sigma <- params[ncol(x)+1]            # Extract the sigma value
  n <- nrow(x)             # Number of data points
  
  # Kernel matrix
  K <- matrix(0, nrow = n, ncol = n)
  for (i in 1:n) {
    for (j in 1:n) {
      K[i, j] <- rbf_ard(x[i, ], x[j, ], length_scales, sigma)
    }
  }
  K <- K + diag(1e-6, n)  # Numerical stability

  # Log marginal likelihood
  L <- chol(K)
  alpha <- solve(t(L), solve(L, y))
  log_likelihood <- -0.5 * t(y) %*% alpha - sum(log(diag(L))) - (n / 2) * log(2 * pi)
  return(-log_likelihood)  # Negative for optimization
}

opt_result_rbf_ard <- optim(
  par = c(rep(1, ncol(x)), sigma = 1),
  fn = gp_model_rbf_ard,
  method = "L-BFGS-B",
  lower = c(rep(0.1, ncol(x)), 0.1),
  upper = c(rep(10, ncol(x)), 10)
)
print(opt_result_rbf_ard$par)

# Print Optimal Parameters
cat("Optimal Parameters:\n")
cat("Length Scales:", opt_result_rbf_ard$par[1:ncol(x)], "\n")
cat("Sigma:", opt_result_rbf_ard$par[ncol(x) + 1], "\n")

# Extract Optimal Parameters
length_scales_opt <- opt_result_rbf_ard$par[1:ncol(x)]  # Extract length scales as a vector
sigma_opt <- opt_result_rbf_ard$par[ncol(x) + 1]        # Extract sigma

# Define ARD RBF Kernel with Optimal Parameters
rbf_ard_with_opt <- function(x1, x2) {
  # Use the optimal parameters directly in the computation
  distance_squared <- sum(((x1 - x2) / length_scales_opt)^2)
  return(sigma_opt^2 * exp(-0.5 * distance_squared))
}
class(rbf_ard_with_opt) <- "kernel"  # Set as kernel class


#' 
#' 
## ----warning=FALSE--------------------------------------------
set.seed(123)

library(Matrix)
library(parallel)

# Optimized ARD RQ Kernel Function
rq_ard <- function(D, alpha, sigma) {
  factor <- 1 + (D / (2 * alpha))
  return(sigma^2 * factor^(-alpha))
}

# Compute Pairwise Squared Distance Matrix Efficiently
compute_distance_matrix <- function(x, length_scales) {
  x_scaled <- sweep(x, 2, length_scales, "/")  # Normalize each feature
  D <- as.matrix(dist(x_scaled)^2)  # Squared Euclidean distance
  return(D)
}

# Optimized GP Model with ARD RQ Kernel
gp_model_rq_ard <- function(params) {
  alpha <- params[1]  # Extract alpha
  length_scales <- params[2:(ncol(x) + 1)]  # Extract vector of length scales
  sigma <- params[ncol(x) + 2]  # Extract sigma
  n <- nrow(x)  # Get number of data points
  
  # Compute Distance Matrix Once
  D <- compute_distance_matrix(x, length_scales)
  
  # Compute Kernel Matrix Using Vectorized Operation
  K <- rq_ard(D, alpha, sigma) + diag(1e-6, n)  # Add jitter for stability
  
  # Cholesky Decomposition
  L <- chol(K)
  alpha_vec <- solve(t(L), solve(L, y))
  
  # Log Marginal Likelihood Computation
  log_likelihood <- -0.5 * sum(y * alpha_vec) - sum(log(diag(L))) - (n / 2) * log(2 * pi)
  return(-log_likelihood)  # Negative for optimization
}

# Run Optimization with Vectorized & Parallelized Code
opt_result_rq_ard <- optim(
  par = c(alpha = 1, rep(1, ncol(x)), sigma = 1),
  fn = gp_model_rq_ard,
  method = "L-BFGS-B",
  lower = c(0.1, rep(0.1, ncol(x)), 0.1),
  upper = c(10, rep(10, ncol(x)), 10)
)

# Print Optimal Parameters
cat("Optimal Parameters:\n")
cat("Alpha:", opt_result_rq_ard$par[1], "\n")
cat("Length Scales:", opt_result_rq_ard$par[2:(ncol(x) + 1)], "\n")
cat("Sigma:", opt_result_rq_ard$par[ncol(x) + 2], "\n")

# Store Optimal Parameters
alpha_opt <- opt_result_rq_ard$par[1]  
length_scales_opt <- opt_result_rq_ard$par[2:(ncol(x) + 1)]  
sigma_opt <- opt_result_rq_ard$par[ncol(x) + 2]  

# Optimized ARD RQ Kernel with Optimal Parameters
rq_ard_with_opt <- function(x1, x2) {
  distance_squared <- sum(((x1 - x2) / length_scales_opt)^2)
  factor <- 1 + (distance_squared / (2 * alpha_opt))
  return(sigma_opt^2 * factor^(-alpha_opt))
}
class(rq_ard_with_opt) <- "kernel"

#' 
#' Model Training and Prediction
## -------------------------------------------------------------
best_kernel <- rbf_iso_with_opt
print(best_kernel) 

gp_model1 <- gausspr(X_train, Y_train, kernel = best_kernel)
y_pred <- predict(gp_model1, X_test)
head(y_pred)


#Evaluation Metrics and UQ Visualizations

rmse <- sqrt(mean((Y_test - y_pred)^2))
mae <- mean(abs(Y_test - y_pred))
mape <- mean(abs((Y_test - y_pred) / Y_test)) * 100

cat("RMSE:", rmse, "\n")
cat("MAE:", mae, "\n")
cat("MAPE:", mape, "%\n")

ggplot(data.frame(Time = 1:length(Y_test), Actual = Y_test, Predicted = y_pred), aes(x = Time)) +
  geom_point(aes(y = Actual, color = "Actual")) +
  geom_point(aes(y = Predicted, color = "Predicted")) +
  theme_minimal()

library(ggplot2)

# Create a combined dataset for visualization
df_train <- data.frame(Time = 1:length(Y_train), Value = Y_train, Type = "Train")
df_test <- data.frame(Time = (length(Y_train) + 1):(length(Y_train) + length(Y_test)), 
                      Value = Y_test, Type = "Test")
df_pred <- data.frame(Time = (length(Y_train) + 1):(length(Y_train) + length(Y_test)), Value = y_pred, Type = "Predicted")

# Combine datasets
df_all <- rbind(df_train, df_test)

# Plot
ggplot() +
  geom_line(data = df_all, aes(x = Time, y = Value, color = Type), size = 1) +  # Line for train + test
  geom_point(data = df_test, aes(x = Time, y = Value, color = "Test"), size = 3) +  # Orange points for test
  geom_point(data = df_pred, aes(x = Time, y = Value, color = "Predicted"), size = 4) +  # Green dot for prediction
  scale_color_manual(values = c("Train" = "black", "Test" = "orange", "Predicted" = "green")) +
  theme_minimal() +
  labs(title = "Test Data Prediction",
       x = "Time",
       y = "Value",
       color = "Legend") +
  theme(legend.position = "bottom")


# Compute Kernel Matrix of Training Data
K_train <- as.matrix(kernelMatrix(best_kernel, X_train))

# Compute Kernel Matrix of Test Data
K_test <- as.matrix(kernelMatrix(best_kernel, X_test, X_train))

# Compute Predictive Variance using Kernel Approximation
predictive_variance <- diag(K_test %*% solve(K_train + diag(nrow(K_train)) * 1e-5) %*% t(K_test))

# Standard deviation as uncertainty measure
y_pred_sd <- sqrt(predictive_variance)

#Uncertainty Quantification (UQ)**
data_plot <- data.frame(
  Time = 1:length(Y_test),
  Actual = Y_test,
  Predicted = y_pred,
  Upper = y_pred + 1.96 * y_pred_sd,
  Lower = y_pred - 1.96 * y_pred_sd
)

# Compute 95% Confidence Interval
data_plot$Upper <- y_pred + 1.96 * y_pred_sd
data_plot$Lower <- y_pred - 1.96 * y_pred_sd

# Plot with Uncertainty
ggplot(data_plot, aes(x = Time)) +
  geom_point(aes(y = Actual, color = "Actual")) +
  geom_point(aes(y = Predicted, color = "Predicted")) +
  geom_ribbon(aes(ymin = Lower, ymax = Upper), alpha = 0.2, fill = "blue") +
  theme_minimal() +
  labs(title = "GP-NAR Forecast with Kernel-Based Uncertainty", y = "centered_CO2", x = "Time")


#Kernel-Based Approximation	uses the covariance matrix of the kernel function	More theoretically sound, adapts to data structure	closer to the true Gaussian Process predictive variance.

#Forecasting Future Values using GPR

set.seed(123)

recursive_forecast <- function(model, X_init, steps) {
  pred_values <- numeric(steps)
  x_current <- as.numeric(X_init)  # Convert initial test input to numeric vector

  for (i in 1:steps) {
    # Predict next step (ensure input is a matrix)
    y_next <- predict(model, matrix(x_current, nrow = 1))

    # Store the prediction
    pred_values[i] <- y_next
    
    # Shift input window (drop oldest and append predicted value)
    x_current <- c(x_current[-1], y_next)  # Maintain as numeric vector
  }
  
  return(pred_values)
}

# Perform recursive multi-step forecasting (e.g., 10 steps ahead)
h_steps_ahead <- 10
forecast_values <- recursive_forecast(gp_model1, as.numeric(X_test[nrow(X_test), ]), h_steps_ahead)
print(forecast_values)


library(ggplot2)

# Create dataframe for visualization
df_actual <- data.frame(Time = 1:length(Y_test), Value = Y_test, Type = "Actual")
df_pred <- data.frame(Time = (max(df_actual$Time) + 1):(max(df_actual$Time) + h_steps_ahead), 
                      Value = forecast_values, Type = "Forecast")

# Combine data
df_all <- rbind(df_actual, df_pred)

# Plot
ggplot(df_all, aes(x = Time, y = Value, color = Type)) +
  geom_line(size = 1) +
  geom_point(size = 3) +
  scale_color_manual(values = c("Actual" = "orange", "Forecast" = "green")) +
  theme_minimal() +
  labs(title = "GP-NAR Forecasting", x = "Time", y = "Value", color = "Legend") +
  theme(legend.position = "bottom")


#' 
#' 
#' 
#' Model Training and Prediction
## -------------------------------------------------------------
best_kernel <- rq_iso_with_opt
print(best_kernel) 

gp_model2 <- gausspr(X_train, Y_train, kernel = best_kernel)
y_pred <- predict(gp_model2, X_test)
head(y_pred)


#Evaluation Metrics and UQ Visualizations

rmse <- sqrt(mean((Y_test - y_pred)^2))
mae <- mean(abs(Y_test - y_pred))
mape <- mean(abs((Y_test - y_pred) / Y_test)) * 100

cat("RMSE:", rmse, "\n")
cat("MAE:", mae, "\n")
cat("MAPE:", mape, "%\n")

ggplot(data.frame(Time = 1:length(Y_test), Actual = Y_test, Predicted = y_pred), aes(x = Time)) +
  geom_point(aes(y = Actual, color = "Actual")) +
  geom_point(aes(y = Predicted, color = "Predicted")) +
  theme_minimal()

library(ggplot2)

# Create a combined dataset for visualization
df_train <- data.frame(Time = 1:length(Y_train), Value = Y_train, Type = "Train")
df_test <- data.frame(Time = (length(Y_train) + 1):(length(Y_train) + length(Y_test)), 
                      Value = Y_test, Type = "Test")
df_pred <- data.frame(Time = (length(Y_train) + 1):(length(Y_train) + length(Y_test)), Value = y_pred, Type = "Predicted")

# Combine datasets
df_all <- rbind(df_train, df_test)

# Plot
ggplot() +
  geom_line(data = df_all, aes(x = Time, y = Value, color = Type), size = 1) +  # Line for train + test
  geom_point(data = df_test, aes(x = Time, y = Value, color = "Test"), size = 3) +  # Orange points for test
  geom_point(data = df_pred, aes(x = Time, y = Value, color = "Predicted"), size = 4) +  # Green dot for prediction
  scale_color_manual(values = c("Train" = "black", "Test" = "orange", "Predicted" = "green")) +
  theme_minimal() +
  labs(title = "Test Data Prediction",
       x = "Time",
       y = "Value",
       color = "Legend") +
  theme(legend.position = "bottom")


# Compute Kernel Matrix of Training Data
K_train <- as.matrix(kernelMatrix(best_kernel, X_train))

# Compute Kernel Matrix of Test Data
K_test <- as.matrix(kernelMatrix(best_kernel, X_test, X_train))

# Compute Predictive Variance using Kernel Approximation
predictive_variance <- diag(K_test %*% solve(K_train + diag(nrow(K_train)) * 1e-5) %*% t(K_test))

# Standard deviation as uncertainty measure
y_pred_sd <- sqrt(predictive_variance)

#Uncertainty Quantification (UQ)**
data_plot <- data.frame(
  Time = 1:length(Y_test),
  Actual = Y_test,
  Predicted = y_pred,
  Upper = y_pred + 1.96 * y_pred_sd,
  Lower = y_pred - 1.96 * y_pred_sd
)

# Compute 95% Confidence Interval
data_plot$Upper <- y_pred + 1.96 * y_pred_sd
data_plot$Lower <- y_pred - 1.96 * y_pred_sd

# Plot with Uncertainty
ggplot(data_plot, aes(x = Time)) +
  geom_point(aes(y = Actual, color = "Actual")) +
  geom_point(aes(y = Predicted, color = "Predicted")) +
  geom_ribbon(aes(ymin = Lower, ymax = Upper), alpha = 0.2, fill = "blue") +
  theme_minimal() +
  labs(title = "GP-NAR Forecast with Kernel-Based Uncertainty", y = "centered_CO2", x = "Time")




#Kernel-Based Approximation	uses the covariance matrix of the kernel function	More theoretically sound, adapts to data structure	closer to the true Gaussian Process predictive variance.

#Forecasting Future Values using GPR


set.seed(123)

recursive_forecast <- function(model, X_init, steps) {
  pred_values <- numeric(steps)
  x_current <- as.numeric(X_init)  # Convert initial test input to numeric vector

  for (i in 1:steps) {
    # Predict next step (ensure input is a matrix)
    y_next <- predict(model, matrix(x_current, nrow = 1))

    # Store the prediction
    pred_values[i] <- y_next
    
    # Shift input window (drop oldest and append predicted value)
    x_current <- c(x_current[-1], y_next)  # Maintain as numeric vector
  }
  
  return(pred_values)
}

# Perform recursive multi-step forecasting (e.g., 10 steps ahead)
h_steps_ahead <- 10
forecast_values <- recursive_forecast(gp_model2, as.numeric(X_test[nrow(X_test), ]), h_steps_ahead)
print(forecast_values)


library(ggplot2)

# Create dataframe for visualization
df_actual <- data.frame(Time = 1:length(Y_test), Value = Y_test, Type = "Actual")
df_pred <- data.frame(Time = (max(df_actual$Time) + 1):(max(df_actual$Time) + h_steps_ahead), 
                      Value = forecast_values, Type = "Forecast")

# Combine data
df_all <- rbind(df_actual, df_pred)

# Plot
ggplot(df_all, aes(x = Time, y = Value, color = Type)) +
  geom_line(size = 1) +
  geom_point(size = 3) +
  scale_color_manual(values = c("Actual" = "orange", "Forecast" = "green")) +
  theme_minimal() +
  labs(title = "GP-NAR Forecasting", x = "Time", y = "Value", color = "Legend") +
  theme(legend.position = "bottom")


#' 
#' 
## -------------------------------------------------------------
best_kernel <- rbf_ard_with_opt
print(best_kernel) 

gp_model3 <- gausspr(X_train, Y_train, kernel = best_kernel)
y_pred <- predict(gp_model3, X_test)
head(y_pred)


#Evaluation Metrics and UQ Visualizations

rmse <- sqrt(mean((Y_test - y_pred)^2))
mae <- mean(abs(Y_test - y_pred))
mape <- mean(abs((Y_test - y_pred) / Y_test)) * 100

cat("RMSE:", rmse, "\n")
cat("MAE:", mae, "\n")
cat("MAPE:", mape, "%\n")

ggplot(data.frame(Time = 1:length(Y_test), Actual = Y_test, Predicted = y_pred), aes(x = Time)) +
  geom_point(aes(y = Actual, color = "Actual")) +
  geom_point(aes(y = Predicted, color = "Predicted")) +
  theme_minimal()

library(ggplot2)

# Create a combined dataset for visualization
df_train <- data.frame(Time = 1:length(Y_train), Value = Y_train, Type = "Train")
df_test <- data.frame(Time = (length(Y_train) + 1):(length(Y_train) + length(Y_test)), 
                      Value = Y_test, Type = "Test")
df_pred <- data.frame(Time = (length(Y_train) + 1):(length(Y_train) + length(Y_test)), Value = y_pred, Type = "Predicted")

# Combine datasets
df_all <- rbind(df_train, df_test)

# Plot
ggplot() +
  geom_line(data = df_all, aes(x = Time, y = Value, color = Type), size = 1) +  # Line for train + test
  geom_point(data = df_test, aes(x = Time, y = Value, color = "Test"), size = 3) +  # Orange points for test
  geom_point(data = df_pred, aes(x = Time, y = Value, color = "Predicted"), size = 4) +  # Green dot for prediction
  scale_color_manual(values = c("Train" = "black", "Test" = "orange", "Predicted" = "green")) +
  theme_minimal() +
  labs(title = "Test Data Prediction",
       x = "Time",
       y = "Value",
       color = "Legend") +
  theme(legend.position = "bottom")


# Compute Kernel Matrix of Training Data
K_train <- as.matrix(kernelMatrix(best_kernel, X_train))

# Compute Kernel Matrix of Test Data
K_test <- as.matrix(kernelMatrix(best_kernel, X_test, X_train))

# Compute Predictive Variance using Kernel Approximation
predictive_variance <- diag(K_test %*% solve(K_train + diag(nrow(K_train)) * 1e-5) %*% t(K_test))

# Standard deviation as uncertainty measure
y_pred_sd <- sqrt(predictive_variance)

#Uncertainty Quantification (UQ)**
data_plot <- data.frame(
  Time = 1:length(Y_test),
  Actual = Y_test,
  Predicted = y_pred,
  Upper = y_pred + 1.96 * y_pred_sd,
  Lower = y_pred - 1.96 * y_pred_sd
)

# Compute 95% Confidence Interval
data_plot$Upper <- y_pred + 1.96 * y_pred_sd
data_plot$Lower <- y_pred - 1.96 * y_pred_sd

# Plot with Uncertainty
ggplot(data_plot, aes(x = Time)) +
  geom_point(aes(y = Actual, color = "Actual")) +
  geom_point(aes(y = Predicted, color = "Predicted")) +
  geom_ribbon(aes(ymin = Lower, ymax = Upper), alpha = 0.2, fill = "blue") +
  theme_minimal() +
  labs(title = "GP-NAR Forecast with Kernel-Based Uncertainty", y = "centered_CO2", x = "Time")


#Kernel-Based Approximation	uses the covariance matrix of the kernel function	More theoretically sound, adapts to data structure	closer to the true Gaussian Process predictive variance.

#Forecasting Future Values using GPR

set.seed(123)

recursive_forecast <- function(model, X_init, steps) {
  pred_values <- numeric(steps)
  x_current <- as.numeric(X_init)  # Convert initial test input to numeric vector

  for (i in 1:steps) {
    # Predict next step (ensure input is a matrix)
    y_next <- predict(model, matrix(x_current, nrow = 1))

    # Store the prediction
    pred_values[i] <- y_next
    
    # Shift input window (drop oldest and append predicted value)
    x_current <- c(x_current[-1], y_next)  # Maintain as numeric vector
  }
  
  return(pred_values)
}

# Perform recursive multi-step forecasting (e.g., 10 steps ahead)
h_steps_ahead <- 10
forecast_values <- recursive_forecast(gp_model3, as.numeric(X_test[nrow(X_test), ]), h_steps_ahead)
print(forecast_values)


library(ggplot2)

# Create dataframe for visualization
df_actual <- data.frame(Time = 1:length(Y_test), Value = Y_test, Type = "Actual")
df_pred <- data.frame(Time = (max(df_actual$Time) + 1):(max(df_actual$Time) + h_steps_ahead), 
                      Value = forecast_values, Type = "Forecast")

# Combine data
df_all <- rbind(df_actual, df_pred)

# Plot
ggplot(df_all, aes(x = Time, y = Value, color = Type)) +
  geom_line(size = 1) +
  geom_point(size = 3) +
  scale_color_manual(values = c("Actual" = "orange", "Forecast" = "green")) +
  theme_minimal() +
  labs(title = "GP-NAR Forecasting", x = "Time", y = "Value", color = "Legend") +
  theme(legend.position = "bottom")


#' 
#' Model Training and Prediction
## -------------------------------------------------------------
best_kernel <- rq_ard_with_opt
print(best_kernel) 

gp_model4 <- gausspr(X_train, Y_train, kernel = best_kernel)
y_pred <- predict(gp_model4, X_test)
head(y_pred)


#Evaluation Metrics and UQ Visualizations

rmse <- sqrt(mean((Y_test - y_pred)^2))
mae <- mean(abs(Y_test - y_pred))
mape <- mean(abs((Y_test - y_pred) / Y_test)) * 100

cat("RMSE:", rmse, "\n")
cat("MAE:", mae, "\n")
cat("MAPE:", mape, "%\n")

ggplot(data.frame(Time = 1:length(Y_test), Actual = Y_test, Predicted = y_pred), aes(x = Time)) +
  geom_point(aes(y = Actual, color = "Actual")) +
  geom_point(aes(y = Predicted, color = "Predicted")) +
  theme_minimal()

library(ggplot2)

# Create a combined dataset for visualization
df_train <- data.frame(Time = 1:length(Y_train), Value = Y_train, Type = "Train")
df_test <- data.frame(Time = (length(Y_train) + 1):(length(Y_train) + length(Y_test)), 
                      Value = Y_test, Type = "Test")
df_pred <- data.frame(Time = (length(Y_train) + 1):(length(Y_train) + length(Y_test)), Value = y_pred, Type = "Predicted")

# Combine datasets
df_all <- rbind(df_train, df_test)

# Plot
ggplot() +
  geom_line(data = df_all, aes(x = Time, y = Value, color = Type), size = 1) +  # Line for train + test
  geom_point(data = df_test, aes(x = Time, y = Value, color = "Test"), size = 3) +  # Orange points for test
  geom_point(data = df_pred, aes(x = Time, y = Value, color = "Predicted"), size = 4) +  # Green dot for prediction
  scale_color_manual(values = c("Train" = "black", "Test" = "orange", "Predicted" = "green")) +
  theme_minimal() +
  labs(title = "Test Data Prediction",
       x = "Time",
       y = "Value",
       color = "Legend") +
  theme(legend.position = "bottom")


# Compute Kernel Matrix of Training Data
K_train <- as.matrix(kernelMatrix(best_kernel, X_train))

# Compute Kernel Matrix of Test Data
K_test <- as.matrix(kernelMatrix(best_kernel, X_test, X_train))

# Compute Predictive Variance using Kernel Approximation
predictive_variance <- diag(K_test %*% solve(K_train + diag(nrow(K_train)) * 1e-5) %*% t(K_test))

# Standard deviation as uncertainty measure
y_pred_sd <- sqrt(predictive_variance)

#Uncertainty Quantification (UQ)**
data_plot <- data.frame(
  Time = 1:length(Y_test),
  Actual = Y_test,
  Predicted = y_pred,
  Upper = y_pred + 1.96 * y_pred_sd,
  Lower = y_pred - 1.96 * y_pred_sd
)

# Compute 95% Confidence Interval
data_plot$Upper <- y_pred + 1.96 * y_pred_sd
data_plot$Lower <- y_pred - 1.96 * y_pred_sd
data_plot$Upper 
data_plot$Lower 

# Plot with Uncertainty
ggplot(data_plot, aes(x = Time)) +
  geom_point(aes(y = Actual, color = "Actual")) +
  geom_point(aes(y = Predicted, color = "Predicted")) +
  geom_ribbon(aes(ymin = Lower, ymax = Upper), alpha = 0.2, fill = "blue") +
  theme_minimal() +
  labs(title = "GP-NAR Forecast with Kernel-Based Uncertainty", y = "centered_CO2", x = "Time")


#Kernel-Based Approximation	uses the covariance matrix of the kernel function	More theoretically sound, adapts to data structure	closer to the true Gaussian Process predictive variance.


#Forecasting Future Values using GPR
set.seed(123)

recursive_forecast <- function(model, X_init, steps) {
  pred_values <- numeric(steps)
  x_current <- as.numeric(X_init)  # Convert initial test input to numeric vector

  for (i in 1:steps) {
    # Predict next step (ensure input is a matrix)
    y_next <- predict(model, matrix(x_current, nrow = 1))

    # Store the prediction
    pred_values[i] <- y_next
    
    # Shift input window (drop oldest and append predicted value)
    x_current <- c(x_current[-1], y_next)  # Maintain as numeric vector
  }
  
  return(pred_values)
}

# Perform recursive multi-step forecasting (e.g., 10 steps ahead)
h_steps_ahead <- 10
forecast_values <- recursive_forecast(gp_model4, as.numeric(X_test[nrow(X_test), ]), h_steps_ahead)
print(forecast_values)


library(ggplot2)

# Create dataframe for visualization
df_actual <- data.frame(Time = 1:length(Y_test), Value = Y_test, Type = "Actual")
df_pred <- data.frame(Time = (max(df_actual$Time) + 1):(max(df_actual$Time) + h_steps_ahead), 
                      Value = forecast_values, Type = "Forecast")

# Combine data
df_all <- rbind(df_actual, df_pred)

# Plot
ggplot(df_all, aes(x = Time, y = Value, color = Type)) +
  geom_line(size = 1) +
  geom_point(size = 3) +
  scale_color_manual(values = c("Actual" = "orange", "Forecast" = "green")) +
  theme_minimal() +
  labs(title = "GP-NAR Forecasting", x = "Time", y = "Value", color = "Legend") +
  theme(legend.position = "bottom")


