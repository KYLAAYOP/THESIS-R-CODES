#' ---
#' title: "EDA"
#' author: "Kyla M. Ayop"
#' date: "`r Sys.Date()`"
#' output:
#'   html_document: default
#'   pdf_document: default
#' ---
#' Certainly! Below are some common R codes for conducting **Exploratory Data Analysis (EDA)** tailored to time series datasets like yours. Each step demonstrates how to explore key aspects of the data, including summary statistics, missing values, trends, seasonality, and visualizations.
#' 
#' ---
## ----warning=FALSE, message=FALSE-----------------------------
# Load libraries
library(tidyverse)
library(lubridate)
library(ggplot2)
library(ggfortify)
library(zoo)
library(corrplot)  # <-- This is essential for using corrplot

# Load CO2 dataset
co2_data <- read.csv("co2_mm_mlo.csv")

# Inspect data
str(co2_data)
summary(co2_data)
head(co2_data)

# Rename columns for ease (if needed)
#colnames(co2_data) <- c("year", "month", "decimal_date", "average", "interpolated", "trend", "num_days")

# Check for missing values
colSums(is.na(co2_data))

# Time series conversion
co2_data$date <- as.Date(paste(co2_data$Year, co2_data$Month, "1", sep = "-"))
co2_ts <- ts(co2_data$Average, start = c(min(co2_data$Year), 3), frequency = 12)

# Plot full CO2 trend
ggplot(co2_data, aes(x = date, y = Average)) +
  geom_line(color = "blue") +
  labs(title = "Monthly CO2 Concentration at Mauna Loa",
       x = "Date", y = "Interpolated CO2 (ppm)")

# Seasonal decompositionM
decomp <- stl(co2_ts, s.window = "periodic")
autoplot(decomp)

# Boxplot by month to show seasonality
co2_data$month <- factor(co2_data$Month, levels = 1:12, labels = month.abb)
ggplot(co2_data, aes(x = month, y = Average)) +
  geom_boxplot(fill = "skyblue") +
  labs(title = "Monthly Distribution of CO2 Levels", x = "Month", y = "Average CO2 (ppm)")
# Rolling mean (12-month)
co2_data$rolling_avg <- zoo::rollmean(co2_data$Average, k = 12, fill = NA)
ggplot(co2_data, aes(x = date)) +
  geom_line(aes(y = Average), color = "gray") +
  geom_line(aes(y = rolling_avg), color = "red") +
  labs(title = "CO2 with 12-Month Rolling Average", y = "CO2 ppm")

# Histogram & Density
ggplot(co2_data, aes(x = Average)) +
  geom_histogram(fill = "lightblue", bins = 30) +
  geom_density(color = "red") +
  labs(title = "Distribution of CO2 Levels", x = "Average CO2")

# Autocorrelation and Partial Autocorrelation
acf(na.omit(co2_data$Average), main = "ACF of CO2")
pacf(na.omit(co2_data$Average), main = "PACF of CO2")

# Correlation matrix (for numeric variables)
cor_matrix <- cor(co2_data %>% select(where(is.numeric)), use = "complete.obs")
corrplot(cor_matrix, method = "circle", title = "Correlation Matrix for CO2 Data")


#' 
## -------------------------------------------------------------
# Load farmgate prices dataset
farm_data <- read.csv("farmgate_prices.csv")

# Inspect structure and summary
str(farm_data)
summary(farm_data)
head(farm_data)

# Check for missing values
colSums(is.na(farm_data))

# Convert date/time if available
# Time series conversion
farm_data$date <- as.Date(paste(farm_data$Year, farm_data$Month, "1", sep = "-"))
farm_ts <- ts(farm_data$Price, start = c(min(farm_data$Year), (farm_data$Month)), frequency = 12)

# Plot full CO2 trend
ggplot(farm_data, aes(x = date, y = Price)) +
  geom_line(color = "blue") +
  labs(title = "Monthly Farmgate Price",
       x = "Date", y = "Price")

# Seasonal decomposition
decomp <- stl(farm_ts, s.window = "periodic")
autoplot(decomp)

# Boxplot by month to show seasonality
farm_data$month <- factor(farm_data$Month, levels = 1:12, labels = month.abb)
ggplot(farm_data, aes(x = month, y = Price)) +
  geom_boxplot(fill = "skyblue") +
  labs(title = "Monthly Farmgate Prices", x = "Month", y = "Price")
# Rolling average
farm_data$rolling_avg <- zoo::rollmean(farm_data$Price, k = 12, fill = NA)
ggplot(farm_data, aes(x = date)) +
  geom_line(aes(y = Price), color = "gray") +
  geom_line(aes(y = rolling_avg), color = "red") +
  labs(title = "Farmgate Prices with 12-Month Rolling Average", y = "Price")

# Distribution
ggplot(farm_data, aes(x = Price)) +
  geom_histogram(fill = "lightblue", bins = 30) +
  geom_density(color = "red") +
  labs(title = "Distribution of Farmgate Prices", x = "Price")

# ACF and PACF
acf(na.omit(farm_data$Price), main = "ACF of Farmgate Prices")
pacf(na.omit(farm_data$Price), main = "PACF of Farmgate Prices")

# Correlation matrix
cor_matrix2 <- cor(farm_data %>% select(where(is.numeric)), use = "complete.obs")
corrplot(cor_matrix2, method = "circle", title = "Correlation Matrix for Farmgate Prices")

#' 
## -------------------------------------------------------------
# Load Vancouver crime dataset
crime_data <- read.csv("vancouver_crime.csv")

# Inspect structure and summary
str(crime_data)
summary(crime_data)
head(crime_data)

# Check for missing values
colSums(is.na(crime_data))

# Convert date/time if available
# Time series conversion
crime_data$date <- as.Date(paste(crime_data$Year, crime_data$Month, "1", sep = "-"))
crime_ts <- ts(crime_data$TfA, start = c(min(crime_data$Year), min(crime_data$Month)), frequency = 12)

# Plot full crime trend
ggplot(crime_data, aes(x = date, y = TfA)) +
  geom_line(color = "blue") +
  labs(title = "Monthly Vancouver Crime Analysis",
       x = "Date", y = "Incidents")

# Seasonal decomposition
decomp <- stl(crime_ts, s.window = "periodic")
autoplot(decomp)

# Boxplot by month to show seasonality
crime_data$month <- factor(crime_data$Month, levels = 1:12, labels = month.abb)
ggplot(crime_data, aes(x = month, y = TfA)) +
  geom_boxplot(fill = "skyblue") +
  labs(title = "Monthly Crime Distribution", x = "Month", y = "Incidents")

# Rolling average
crime_data$rolling_avg <- zoo::rollmean(crime_data$TfA, k = 12, fill = NA)
ggplot(crime_data, aes(x = date)) +
  geom_line(aes(y = TfA), color = "gray") +
  geom_line(aes(y = rolling_avg), color = "red") +
  labs(title = "Crime Trends with 12-Month Rolling Average", y = "Incidents")

# Distribution
ggplot(crime_data, aes(x = TfA)) +
  geom_histogram(fill = "lightblue", bins = 30) +
  geom_density(color = "red") +
  labs(title = "Distribution of Vancouver Crime Incidents", x = "Incidents")

# ACF and PACF
acf(na.omit(crime_data$TfA), main = "ACF of Vancouver Crime Incidents")
pacf(na.omit(crime_data$TfA), main = "PACF of Vancouver Crime Incidents")

# Correlation matrix
cor_matrix2 <- cor(crime_data %>% select(where(is.numeric)), use = "complete.obs")
corrplot(cor_matrix2, method = "circle", title = "Correlation Matrix for Vancouver Crime Data")

#' 
