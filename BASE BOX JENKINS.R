#' ---
#' title: "BOX JENKINS"
#' author: "Kyla M. Ayop"
#' date: "`r Sys.Date()`"
#' output: html_document
#' ---
#' 
## ----warning=FALSE,message=FALSE------------------------------
# Load packages
library(forecast)
library(tseries)
library(ggplot2)
library(Metrics)
library(dplyr)
library(gridExtra)

# Function to run full Box-Jenkins pipeline
run_box_jenkins <- function(ts_data, name = "Dataset") {
  cat("===========", name, "===========\n")
  
  # 1. Plot original series
  autoplot(ts_data) + ggtitle(paste(name, "- Original Time Series"))

  # 2. Check stationarity with ADF test
  adf_test <- adf.test(ts_data)
  print(adf_test)
  
  # 3. Differencing if needed
  ndiffs_needed <- ndiffs(ts_data)
  cat("Number of differences needed:", ndiffs_needed, "\n")
  
  ts_diff <- diff(ts_data, differences = ndiffs_needed)
  autoplot(ts_diff) + ggtitle(paste(name, "- Differenced Series"))
  
  # 3.1 Plot differenced series
  p_diff <- autoplot(ts_diff) + ggtitle(paste(name, "- Differenced Series")) + theme_minimal()
  
  # 3.2 Plot ACF & PACF
  p_acf <- ggAcf(ts_diff) + ggtitle("ACF Plot")
  p_pacf <- ggPacf(ts_diff) + ggtitle("PACF Plot")
  grid.arrange(p_diff, p_acf, p_pacf, ncol = 1)
  
  
  # 4. Train-test split (80/20)
  n <- length(ts_data)
  train_end <- floor(0.8 * n)
  train <- window(ts_data, end = time(ts_data)[train_end])
  test <- window(ts_data, start = time(ts_data)[train_end + 1])
  
  # 5. Auto ARIMA
  auto_fit <- auto.arima(train)
  summary(auto_fit)
  
  # 📌 Print the selected ARIMA model structure
  order_vals <- arimaorder(auto_fit)
  cat(sprintf("Selected ARIMA model: ARIMA(%d,%d,%d)(%d,%d,%d)[%d]\n",
              order_vals[1], order_vals[2], order_vals[3],
              order_vals[4], order_vals[5], order_vals[6],
              frequency(train)))
  
  # 6. Manual ARIMA (using output from auto.arima or guessing)
  #manual_fit <- Arima(train, order = c(1, ndiffs_needed, 1))
  
  # 7. Forecasting next point (1-step ahead)
  point_forecast <- forecast(auto_fit, h = 1)

  # 7.1 Evaluate point forecast (1-step ahead)
  actual_point <- test[1]  # First actual value after the training set
  predicted_point <- point_forecast$mean[1]

  # 8. Forecasting for test set
  h_test <- length(test)
  auto_forecast <- forecast(auto_fit, h = h_test)
  #manual_forecast <- forecast(manual_fit, h = h_test)
  
  # 9. Evaluation
  eval_metrics <- function(actual, predicted) {
    data.frame(
      MAE = mae(actual, predicted),
      MAPE = mape(actual, predicted),
      RMSE = rmse(actual, predicted)
    )
  }
  
  auto_metrics <- eval_metrics(test, auto_forecast$mean)
  #manual_metrics <- eval_metrics(test, manual_forecast$mean)
  point_metrics <- eval_metrics(actual_point, predicted_point)

  cat("Point Forecast Metrics (1-step-ahead):\n")
  print(point_metrics)

  cat("Auto ARIMA metrics::\n")
  print(auto_metrics)
  
  #print("Manual ARIMA metrics:")
  #print(manual_metrics)
  
  # 10. Visualization
  p1 <- autoplot(auto_forecast) +
    autolayer(test, series = "Test") +
    ggtitle(paste(name, "- Auto ARIMA Forecast")) +
    theme_minimal()

  # Create a small time series for actual and predicted to plot
  point_time <- time(test)[1]  # Get the time for the forecast point
  
  # Create a data frame for plotting
  df_points <- data.frame(
    Time = c(point_time, point_time),
    Value = c(point_forecast$mean[1], actual_point),
    Type = c("Forecast", "Actual")
  )
  
  # Plot both points
  p2 <- ggplot(df_points, aes(x = Time, y = Value, color = Type)) +
    geom_point(size = 4) +
    ggtitle(paste(name, "- 1-Step-Ahead Forecast Point")) +
    ylab("Value") +
    xlab("Time") +
    theme_minimal() +
    scale_color_manual(values = c("Forecast" = "blue", "Actual" = "black"))

    
  grid.arrange(p1 , p2,ncol = 2)
}

# Load datasets
co2 <- read.csv("co2_mm_mlo.csv")
crime <- read.csv("vancouver_crime.csv")
farm <- read.csv("farmgate_prices.csv")

# Convert to time series
co2_ts <- ts(co2$Average, start = c(min(co2$Year), 3), frequency = 12)
crime_ts <- ts(crime$TfA, start = c(min(crime$Year), min(crime$Month)), frequency = 12)
farm_ts <- ts(farm$Price, start = c(min(farm$Year), min(farm$Month)), frequency = 12)

# Run Box-Jenkins modeling on each dataset
run_box_jenkins(co2_ts, "CO2 Dataset")
run_box_jenkins(crime_ts, "Crime Dataset")
run_box_jenkins(farm_ts, "Farmgate Dataset")


