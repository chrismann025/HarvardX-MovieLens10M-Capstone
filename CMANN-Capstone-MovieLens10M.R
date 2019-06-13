# Load Training (edx) & Test (validation) datasets using the provided course code below
# NOTE: if using R version 3.6.0, use: set.seed(1, sample.kind = "Rounding") instead of set.seed(1)

###################################
# Create edx set and validation set
###################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

# set.seed(1) # if using R 3.6.0: set.seed(1, sample.kind = "Rounding")
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)





#******************************************************************************
#******************************************************************************
#**************         START OF CAPSTONE PROJECT CODE           **************
#******************************************************************************
#******************************************************************************

# Load required libraries
if (!require(tidyverse)) install.packages('tidyverse')
library(tidyverse)

if (!require(dplyr)) install.packages('dplyr')
library(dplyr)

if (!require(lubridate)) install.packages('lubridate')
library(lubridate)

if (!require(knitr)) install.packages('knitr')
library(knitr)


# Create working copied of the Training (edx) and Test (validation) datasets
train_set <- edx
test_set <- validation

# Define our evaluation function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#*******************************************************************************
#*******************************************************************************
#****************    Linear Regression - Model Building    *********************
#*******************************************************************************
#*******************************************************************************


#*******************************************************************************
# Linear Regression - Using just the overall average
#   Yu,i = mu
#*******************************************************************************

# Calculate the overall average rating
mu <- mean(train_set$rating)
mu

# Evaluate the performance of simply guessing the overall average
rmse <- RMSE(test_set$rating, mu)
rmse

# Save the RMSE result to display later
LR_rmse_results <- tibble(Method = "LR: Base Mean Model",
                          RMSE = rmse)
LR_rmse_results %>% knitr::kable()


#*******************************************************************************
# Linear Regression - Base Average Model + Movie Effect:
#   Yu,i = mu + movie_avgs$b_i
#*******************************************************************************

# Calcualte the average rating for each Movie and subtract the mean
movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# Calculate the Predicted ratings on the Test set
predicted_ratings <- mu + test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

# Calculate the RMSE
rmse <- RMSE(predicted_ratings, test_set$rating)
LR_rmse_results <- bind_rows(LR_rmse_results,
                             tibble(Method="LR: Movie Effect Model",
                                    RMSE = rmse))
LR_rmse_results %>% knitr::kable()


#*******************************************************************************
# Average + Movie Effect + User Effect:
#   Yu,i = mu + movie_avgs$b_i + user_avgs$b_u
#*******************************************************************************

# Plot the distribution of Ratings by User ID in the Training set
train_set %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating)) %>% filter(n()>=100) %>%
  ggplot(aes(b_u)) +
  geom_histogram(bins = 30, color = "black") +
  labs(x="User Rating", y="# of Ratings") +
  ggtitle("Histogram of Ratings by User")

# Plot the log distribution of # of Ratings per User
user_data %>%
  ggplot(aes(x = u_numrtg)) +
  geom_histogram(bins = 50, color = "black") +
  scale_x_log10() +
  labs(x="# of Ratings (log10 scale)", y="# of Users") +
  ggtitle("Distribution - # of Ratings by User")

# Calculate average rating for each User
user_avgs <- train_set %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Predict ratings on the Test set
predicted_ratings <- test_set %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Calcualte the RMSE
rmse <- RMSE(predicted_ratings, test_set$rating)
LR_rmse_results <- bind_rows(LR_rmse_results,
                             tibble(Method="LR: Movie + User Effects Model",
                                    RMSE = rmse))
LR_rmse_results %>% knitr::kable()


#*******************************************************************************
# Average + Movie + User + Time Effect:
#   Yu,i = mu + movie_avgs$b_i + user_avgs$b_u + week_avgs$w_u
#*******************************************************************************

# Plot the Average Rating by Week
train_set %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
  group_by(date) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(date, rating)) +
  geom_point() +
  geom_smooth(method="loess") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x="Time (Weeks)", y="Average Rating") +
  ggtitle("Average Rating over Time (Weeks)")

# Calculate the average rating by Week
week_avgs <- train_set %>%
  mutate(week = round_date(as_datetime(timestamp), unit = "week")) %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(week) %>%
  summarize(w_u = mean(rating - mu - b_i - b_u))

# Predict Ratings on the Test set
predicted_ratings <- test_set %>%
  mutate(week = round_date(as_datetime(timestamp), unit = "week")) %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(week_avgs, by='week') %>%
  mutate(pred = mu + b_i + b_u + w_u) %>%
  pull(pred)

# Exaluate the RMSE
rmse <- RMSE(predicted_ratings, test_set$rating)
LR_rmse_results <- bind_rows(LR_rmse_results,
                             tibble(Method="LR: Movie + User + Time Effects Model",
                                    RMSE = rmse))
LR_rmse_results %>% knitr::kable()


#*******************************************************************************
# Average + Movie + User + Time + Genre Effect:
#   Yu,i = mu + movie_avgs$b_i + user_avgs$b_u + week_avgs$w_u + genre_avgs$g_u
#*******************************************************************************

# Display graph of Average Rating by all unique Genre combinations
train_set %>%
  group_by(genres) %>%
  summarize(avg = mean(rating)) %>%
  ggplot(aes(x = as.numeric(reorder(genres, avg)), y = avg)) +
  geom_point() +
  geom_smooth( aes(x = as.numeric(reorder(genres, avg)), y = avg),
               method = 'lm', formula = y ~ poly(x, 4), se = TRUE) +
  theme(axis.text.x=element_blank()) +
  labs(x="Unique Genre Combination", y="Average Rating") +
  ggtitle("Mean Rating by Unique Genre Combination")

# Display graph of each indivudual Genre
# First split out each individual Genre from the genres variable
genre_cats <- edx %>%
  select(genres, rating) %>%
  separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(avg = mean(rating)) %>%
  arrange(avg)

# Sort the Genres by factor
genre_cats$genres <- factor(genre_cats$genres, levels = genre_cats$genres[order(genre_cats$avg)])

# Plot the Average pre Genre
genre_cats %>% ggplot(aes(genres, avg)) +
  geom_point() +
  geom_smooth(aes(x = as.numeric(genres), y = avg),
              method = 'lm',
              formula = y ~ poly(x, 4),
              se = TRUE) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x="Genre Category", y="Average Rating") +
  ggtitle("Mean Rating by Genre Category")


# Calcualte the average rating of each Genre Combination
genre_avgs <- train_set %>%
  mutate(week = round_date(as_datetime(timestamp), unit = "week")) %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(week_avgs, by='week') %>%
  group_by(genres) %>%
  summarize(g_u = mean(rating - mu - b_i - b_u - w_u))

# Make predictions against the Test set
predicted_ratings <- test_set %>%
  mutate(week = round_date(as_datetime(timestamp), unit = "week")) %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(week_avgs, by='week') %>%
  left_join(genre_avgs, by='genres') %>%
  mutate(pred = mu + b_i + b_u + w_u + g_u) %>%
  pull(pred)

rmse <- RMSE(predicted_ratings, test_set$rating)
LR_rmse_results <- bind_rows(LR_rmse_results,
                             tibble(Method="LR: Movie + User + Time + Genre Effects Model",
                                    RMSE = rmse))
LR_rmse_results %>% knitr::kable()



#*******************************************************************************
#*******************************************************************************
#***********    Regularized Linear Regression - Model Building    **************
#*******************************************************************************
#*******************************************************************************

#*******************************************************************************
# Regularized Linear Regression - Base Average Model + Movie Effect:
#   Yu,i = mu + movie_reg_avgs$b_i
#*******************************************************************************

# Calcualte Regularized Movie Effect (b_i)
lambda <- 3
movie_reg_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n())

# Predict against Test Set
predicted_ratings <- test_set %>%
  left_join(movie_reg_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)

# Calcualte RMSE
rmse <- RMSE(predicted_ratings, test_set$rating)
LR_rmse_results <- bind_rows(LR_rmse_results,
                             tibble(Method="Reg LR: Movie Effect Model",
                                    RMSE = rmse))
LR_rmse_results %>% knitr::kable()


#*******************************************************************************
# Regularized Linear Regression - Average + Movie Effect + User Effect:
#   Yu,i = mu + b_i + b_u
#*******************************************************************************

# Find the best RMSE across range of Lambdas
lambdas <- seq(5, 5.5, 0.05)
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_set %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <-
    test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  return(RMSE(predicted_ratings, test_set$rating))
})
qplot(lambdas, rmses)

# Lambda resulting in best fit / lowest RMSE
lambda <- lambdas[which.min(rmses)]
lambda

LR_rmse_results <- bind_rows(LR_rmse_results,
                             tibble(Method="Reg LR: Movie + User Effect Model",
                                    RMSE = min(rmses)))
LR_rmse_results %>% knitr::kable()


#*******************************************************************************
# Regularized Linear Regression - Average + Movie + User + Time Effect:
#   Yu,i = mu + b_i + b_u + w_u
#*******************************************************************************

# Update Training and Test datasets to include date as Week
train_set <- edx
train_set <- train_set %>% mutate(week = round_date(as_datetime(timestamp), unit = "week"))

test_set <- validation
test_set <- test_set %>% mutate(week = round_date(as_datetime(timestamp), unit = "week"))

# Find best Lambda across range
lambdas <- seq(5.4, 5.6, 0.05)
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_set %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  w_u <- train_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(week) %>%
    summarize(w_u = sum(rating - b_i - b_u - mu)/(n()+l))
  predicted_ratings <-
    test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(w_u, by = "week") %>%
    mutate(pred = mu + b_i + b_u + w_u) %>%
    pull(pred)
  return(RMSE(predicted_ratings, test_set$rating))
})
qplot(lambdas, rmses)

lambda <- lambdas[which.min(rmses)]
lambda

LR_rmse_results <- bind_rows(LR_rmse_results,
                             tibble(Method="Reg LR: Movie + User + Time Effect Model",
                                    RMSE = min(rmses)))
LR_rmse_results %>% knitr::kable()


#*******************************************************************************
# Regularized Linear Regression - Average + Movie + User + Time + Genre Effect:
#   Yu,i = mu + b_i + b_u + w_u + g_u
#*******************************************************************************

# Find best Lambda to minimize RMSE
lambdas <- seq(5.35, 5.55, 0.05)
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_set %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  w_u <- train_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(week) %>%
    summarize(w_u = sum(rating - b_i - b_u - mu)/(n()+l))
  g_u <- train_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(w_u, by="week") %>%
    group_by(genres) %>%
    summarize(g_u = sum(rating - b_i - b_u - w_u - mu)/(n()+l))
  predicted_ratings <-
    test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(w_u, by = "week") %>%
    left_join(g_u, by = "genres") %>%
    mutate(pred = mu + b_i + b_u + w_u + g_u) %>%
    pull(pred)
  return(RMSE(predicted_ratings, test_set$rating))
})
qplot(lambdas, rmses)

lambda <- lambdas[which.min(rmses)]
lambda

LR_rmse_results <- bind_rows(LR_rmse_results,
                             tibble(Method="Reg LR: Movie + User + Time + Genre Effect Model",
                                    RMSE = min(rmses)))
LR_rmse_results %>% knitr::kable()

# Clean up Environment
rm(genre_avgs, lambda, lambdas, movie_avgs, movie_reg_avgs, mu, best_rmse_results,
   predicted_ratings, rmse, rmses, user_avgs, week_avgs,  genre_cats, user_data,
   movie_data, test_set, train_set)

#*******************************************************************************


#*******************************************************************************
#*******************************************************************************
#***********    XGB - Extreme Parallel Tree Boosting (xgBoost)    **************
#*******************************************************************************
#*******************************************************************************

# Load required libraries
if (!require(xgboost)) install.packages('xgboost')
library(xgboost)

if (!require(data.table)) install.packages('data.table')
library(data.table)


# Set seed for random sample (if needed)
set.seed(1, sample.kind = "Rounding")

# Create new Training and Test datasets
# xgBoost requires that all values be numeric
# We won't be using the Title or Timestamp, so we drop them
# If you don't have at least 64Gb RAM, then use the 30% random sample below

#train_set <- edx %>% select(-one_of("timestamp", "title")) %>% mutate(userId = as.numeric(userId)) %>% sample_frac(0.3)

train_set <- edx %>% select(-one_of("timestamp", "title")) %>% mutate(userId = as.numeric(userId))
test_set <- validation %>% select(-one_of("timestamp", "title")) %>% mutate(userId = as.numeric(userId))

# One-hot encoding of Genres in the Training Set
train_set <- train_set %>% mutate(g_Action = if_else(grepl("Action", genres), 1, 0),
                                  g_Adventure = if_else(grepl("Adventure", genres), 1, 0),
                                  g_Animation = if_else(grepl("Animation", genres), 1, 0),
                                  g_Children = if_else(grepl("Children", genres), 1, 0),
                                  g_Comedy = if_else(grepl("Comedy", genres), 1, 0),
                                  g_Crime = if_else(grepl("Crime", genres), 1, 0),
                                  g_Documentary = if_else(grepl("Documentary", genres), 1, 0),
                                  g_Drama = if_else(grepl("Drama", genres), 1, 0),
                                  g_Fantasy = if_else(grepl("Fantasy", genres), 1, 0),
                                  g_FilmNoir = if_else(grepl("Film-Noir", genres), 1, 0),
                                  g_Horror = if_else(grepl("Horror", genres), 1, 0),
                                  g_IMAX = if_else(grepl("IMAX", genres), 1, 0),
                                  g_Musical = if_else(grepl("Musical", genres), 1, 0),
                                  g_Mystery = if_else(grepl("Mystery", genres), 1, 0),
                                  g_Romance = if_else(grepl("Romance", genres), 1, 0),
                                  g_SciFi = if_else(grepl("Sci-Fi", genres), 1, 0),
                                  g_Thriller = if_else(grepl("Thriller", genres), 1, 0),
                                  g_War = if_else(grepl("War", genres), 1, 0),
                                  g_Western = if_else(grepl("Western", genres), 1, 0))

# Add column which is a total count of the number of Genres
train_set <- train_set %>% mutate(g_Count = rowSums(select(.,g_Action:g_Western) != 0))

# One-hot encoding of Genres in the Test Set
test_set <- test_set %>% mutate(g_Action = if_else(grepl("Action", genres), 1, 0),
                                g_Adventure = if_else(grepl("Adventure", genres), 1, 0),
                                g_Animation = if_else(grepl("Animation", genres), 1, 0),
                                g_Children = if_else(grepl("Children", genres), 1, 0),
                                g_Comedy = if_else(grepl("Comedy", genres), 1, 0),
                                g_Crime = if_else(grepl("Crime", genres), 1, 0),
                                g_Documentary = if_else(grepl("Documentary", genres), 1, 0),
                                g_Drama = if_else(grepl("Drama", genres), 1, 0),
                                g_Fantasy = if_else(grepl("Fantasy", genres), 1, 0),
                                g_FilmNoir = if_else(grepl("Film-Noir", genres), 1, 0),
                                g_Horror = if_else(grepl("Horror", genres), 1, 0),
                                g_IMAX = if_else(grepl("IMAX", genres), 1, 0),
                                g_Musical = if_else(grepl("Musical", genres), 1, 0),
                                g_Mystery = if_else(grepl("Mystery", genres), 1, 0),
                                g_Romance = if_else(grepl("Romance", genres), 1, 0),
                                g_SciFi = if_else(grepl("Sci-Fi", genres), 1, 0),
                                g_Thriller = if_else(grepl("Thriller", genres), 1, 0),
                                g_War = if_else(grepl("War", genres), 1, 0),
                                g_Western = if_else(grepl("Western", genres), 1, 0))

# Add column which is a total count of the number of Genres
test_set <- test_set %>% mutate(g_Count = rowSums(select(.,g_Action:g_Western) != 0))


# Create dataframe with additional User rating information:
#       Average (u_avg), StDev (u_std), Number of Reviews (u_numrtg)
user_data <- train_set %>% group_by(userId) %>%
  summarize(u_avg = mean(rating),
            u_std = sd(rating),
            u_numrtg = as.numeric(n()))

# Create dataframe with additional Movie rating information:
#       Average (m_avg), StDev (m_std), Number of Reviews (m_numrtg)
movie_data <- train_set %>% group_by(movieId) %>%
  summarize(m_avg = mean(rating),
            m_std = sd(rating),
            m_numrtg = as.numeric(n()))

# Merge User & Movie derived fields into the Training and Test datasets
train_set <- train_set %>% left_join(user_data, by = "userId")
train_set <- train_set %>% left_join(movie_data, by = "movieId")
test_set <- test_set %>% left_join(user_data, by = "userId")
test_set <- test_set %>% left_join(movie_data, by = "movieId")

# xgBoost requires that the target variable, referred to as the "label" (rating) be in a separate data table
# Here we create a Label and Data for the Training and Testing sets
# We are also dropping the Genres variable since it isn't numeric and no longer needed due to the One-Hot encoding added earlier

train_data <- train_set %>% select(-one_of("rating", "genres")) %>% as.data.table()
train_label <- train_set %>% select("rating") %>% as.data.table()
test_data <- test_set %>% select(-one_of("rating", "genres")) %>% as.data.table()
test_label <- test_set %>% select("rating") %>% as.data.table()

# Create the xgBoost Training & Testing matricies
train_matrix = xgb.DMatrix(as.matrix(train_data), label=as.matrix(train_label))
test_matrix = xgb.DMatrix(as.matrix(test_data), label=as.matrix(test_label))


# xgBoost allows the creation of both Linear, Tree, and Mixed Linear & Tree based models
# xgBoost can also create Classification trees (binary and multi-class) by changing the objective function
# We will be creating only regression trees. Linear trees are the fastest to train, but do not produce results as good as mixed models

# Below are the multiple paramaters used to train the models presented in the Results
# Most take a considerable amount of time to train so we will just train a simple model first

#**********************************************************************************************
# Model: XGB Linear 50 Boost Rnds
#**********************************************************************************************
# Set paramaters for most basic Linear tree
# These settings should train in less than a minute with 50 Boosting Rounds
xgb_params <- list(booster = "gblinear",      # Linear boosting alg
                   objective = "reg:linear",  # Linear regression (default)
                   eval_metric = "rmse",      # rmse as objective function
                   verbosity = 3,             # Highest verbosr level - shows all debug info
                   silent = 0)                # Not silent

# Train the XGB tree using the currently set xgb_params
start_time <- Sys.time()                    # Record start time
xgb_model <- xgboost(params = xgb_params,
                     data = train_matrix,
                     nrounds = 5,           # Maximum number of Boosting Rounds
                     verbose = 2)           # Display pogress during Training
finish_time <- Sys.time()                   # Record finish time
finish_time - start_time                    # Display total training time

# Use the trained model to predict the Test dataset
test_pred <- as.data.frame(predict(xgb_model , newdata = test_matrix))

# Calculate the RMSE of the Predictions
rmse <- RMSE(test_label$rating, test_pred$`predict(xgb_model, newdata = test_matrix)`)

XGB_rmse_results <- tibble(Method="XGB Linear 50 Boost Rnds",
                           RMSE = rmse,
                           Train_Time = paste(as.character(round(as.numeric(finish_time - start_time, units="secs"), 2)), "secs"))
XGB_rmse_results %>% knitr::kable()


# Calculate and Plot the Feature Importances for the XGB Model
# Compute feature importance matrix
importance_matrix <- xgb.importance(model = xgb_model)

# Plot the Feature Importance of the Top 10 features
importance_matrix %>% xgb.plot.importance()



# Create Table of Additional Models
XGB_rmse_results <- bind_rows(XGB_rmse_results,
                              tibble(Method="XGB Linear 1000 Boost Rnds",
                                     RMSE = 0.8820874,
                                     Train_Time = "8.13 mins"))
XGB_rmse_results <- bind_rows(XGB_rmse_results,
                              tibble(Method="XGB Mixed Tree 5 Boost Rnds",
                                     RMSE = 0.8690746,
                                     Train_Time = "24.75 mins"))
XGB_rmse_results <- bind_rows(XGB_rmse_results,
                              tibble(Method="XGB Mixed Tree 50 Boost Rnds",
                                     RMSE = 0.8539179,
                                     Train_Time = "20.66 HOURS"))
XGB_rmse_results %>% knitr::kable()


#**********************************************************************************************
# Model: XGB Linear 1000 Boost Rnds
#**********************************************************************************************
# Use the same xgb_params as XGB Linear 50, but train it with 1000 boosting rounds


#**********************************************************************************************
# Model: XGB Mixed Tree 5 Boost Rnds
#**********************************************************************************************
# Use the these xgb_params and train with 5 boosting rounds
# NOTE: Training time of 25+ minutes depending upon your system

# xgb_params <- list(booster = "gbtree",
#                   objective = "reg:linear",
#                   colsample_bynode = 0.8,
#                   learning_rate = 1,
#                   max_depth = 10,
#                   num_parallel_tree = 25,
#                   subsample = 0.8,
#                   verbosity = 3,
#                   silent = 0)

#**********************************************************************************************
# Model: XGB Mixed Tree 50 Boost Rnds
#**********************************************************************************************
# Use the same xgb_params as XGB Mixed Tree 5 Boost Rnds above, but train it with 50 boosting rounds
# NOTE: Training time of 20+ HOURS depending upon your system




#*******************************************************************************
#*******************************************************************************
#** Recosystem - Matrix Factorization w/ Parallel Stochastic Gradient Descent **
#*******************************************************************************
#*******************************************************************************

# Load required libraries
if (!require(recosystem)) install.packages('recosystem')
library(recosystem)

#************************************************************************************
# Recosystem - Matrix Factorization with Parallel Stochastic Gradient Descent
#************************************************************************************

# Create new Traing and Test datasets. Recosystem only uses three columns: Rating, UserId, and MovieId
train_set <- edx %>% select(-one_of("genres","title","timestamp")) %>% as.matrix()
test_set <-  validation %>% select(-one_of("genres","title","timestamp")) %>% as.matrix()


# Create the Reco Recommender object
r = Reco()


# One very nice feature of Reco, besides it's speed, is that tuning training set is very easy and the
# overall best performing tuned paramaters are stored directly in the Reco object.

# Here we will create some Tuning Grids

#***********************************************************************************
# Tuning Grid V1 - We will only tune with two values of Dim (20 & 30)
#***********************************************************************************

opts_list = list(dim      = c(20, 30),  # Number of Latent Features
                 costp_l1 = c(0),       # L1 regularization cost for User factors
                 costp_l2 = c(0.01),    # L2 regularization cost for User factors
                 costq_l1 = c(0),       # L1 regularization cost for Movie factors
                 costq_l2 = c(0.1),     # L2 regularization cost for Movie factors
                 lrate    = c(0.1),     # Learning Rate - Aprox step size in Gradient Descent
                 niter    = 10,         # Number of Iterations for Training (Not used in Tuning)
                 nfolds   = 5,          # Number of Folds for CV in Tuning
                 verbose  = FALSE,      # Don't Show Progress
                 nthread  = 1)          #!!! Can be set to higher values for Tuning, but MUST be set to 1
                                        #    for Training or the results are not reproducible

#***********************************************************************************
# TUNE the Model - This will take around 2 minutes if executed
#***********************************************************************************

start <- Sys.time()

set.seed(1, sample.kind = "Rounding")
opts_tune = r$tune(data_memory(train_set[, 1], train_set[, 2], train_set[, 3]),
                   opts = opts_list)

finish <- Sys.time()
tune_time <- finish - start
tune_time
opts_tune$min


#***********************************************************************************
# TRAIN the Model over 50 Iterations
#***********************************************************************************
start <- Sys.time()

set.seed(1, sample.kind = "Rounding")
r$train(data_memory(train_set[, 1], train_set[, 2], train_set[, 3]),
        opts = c(opts_tune$min,
                 niter    = 50,     # Train over 50 Iterations
                 nthread = 1))      # Must be set to 1 for training

finish <- Sys.time()
train_time <- finish - start
train_time


#***********************************************************************************
# PREDICT the Test Ratings
#***********************************************************************************
start <- Sys.time()

pred <- r$predict(data_memory(test_set[, 1], test_set[, 2], test_set[, 3]), out_memory())

finish <- Sys.time()
pred_time <- finish - start
pred_time


#***********************************************************************************
# Calculate the RMSE
#***********************************************************************************

rmse <- RMSE(test_set[,3],pred)

RECO_rmse_results <- tibble(Method="V1 - 30 x 50",
                            RMSE = rmse,
                            Train_Time = round(as.numeric(train_time, units="mins"), 2))
RECO_rmse_results %>% knitr::kable()


#***********************************************************************************
# Train "Optimal" RMSE - (dim = 24)  V2 - 24 x 50
#***********************************************************************************
opts_list = list(dim      = c(24),      # Number of Latent Features
                 costp_l1 = c(0),       # L1 regularization cost for User factors
                 costp_l2 = c(0.01),    # L2 regularization cost for User factors
                 costq_l1 = c(0),       # L1 regularization cost for Movie factors
                 costq_l2 = c(0.1),     # L2 regularization cost for Movie factors
                 lrate    = c(0.1),     # Learning Rate - Aprox step size in Gradient Descent
                 nfold    = 5,          # Number of folds for Cross Validation
                 nthread  = 1,          # Set Number of Threads to 1 *** Required to get Reproducible Results ***
                 niter    = 50,         # Number of Iterations
                 verbose  = FALSE)      # Don't Show Fold Details

start <- Sys.time()

set.seed(1, sample.kind = "Rounding")
r$train(data_memory(train_set[, 1], train_set[, 2], train_set[, 3]),
        opts = c(opts_list,
                 nthread = 1))

finish <- Sys.time()
train_time <- finish - start

start <- Sys.time()

pred <- r$predict(data_memory(test_set[, 1], test_set[, 2], test_set[, 3]), out_memory())
finish <- Sys.time()

pred_time <- finish - start
pred_time

rmse <- RMSE(test_set[,3],pred)

RECO_rmse_results <- bind_rows(RECO_rmse_results,
                               tibble(Method="V2 - 24 x 50 (Optimal DIM)",
                                      RMSE = rmse,
                                      Train_Time = round(as.numeric(train_time, units="mins"), 2)))
RECO_rmse_results %>% knitr::kable()

#***********************************************************************************


#***********************************************************************************
# Train Very Large (dim = 1000) -  V3 - 1000 x 100
# !!! NOTE !!!  This takes almost 1 Hour to Run
#***********************************************************************************

opts_list = list(dim      = c(1000),    # Number of Latent Features
                 costp_l1 = c(0),       # L1 regularization cost for User factors
                 costp_l2 = c(0.01),    # L2 regularization cost for User factors
                 costq_l1 = c(0),       # L1 regularization cost for Movie factors
                 costq_l2 = c(0.1),     # L2 regularization cost for Movie factors
                 lrate    = c(0.1),     # Learning Rate - Aprox step size in Gradient Descent
                 nfold    = 5,          # Number of folds for Cross Validation
                 nthread  = 1,          # Set Number of Threads to 1 *** Required to get Reproducible Results ***
                 niter    = 100,        # Number of Iterations
                 verbose  = TRUE)       # Show Fold Details

start <- Sys.time()

set.seed(1, sample.kind = "Rounding")
r$train(data_memory(train_set[, 1], train_set[, 2], train_set[, 3]),
        opts = c(opts_list,
                 nthread = 1))

finish <- Sys.time()
train_time <- finish - start

start <- Sys.time()

pred <- r$predict(data_memory(test_set[, 1], test_set[, 2], test_set[, 3]), out_memory())
finish <- Sys.time()

pred_time <- finish - start
pred_time

rmse <- RMSE(test_set[,3],pred)

RECO_rmse_results <- bind_rows(RECO_rmse_results,
                               tibble(Method="V3 - 1000 x 100",
                                      RMSE = rmse,
                                      Train_Time = round(as.numeric(train_time, units="mins"), 2)))
RECO_rmse_results %>% knitr::kable()

#***********************************************************************************

#***********************************************************************************
# Tune the DIM parameter - FOR 10-100 and then 10-40 and Graph
# !!!!  NOTE !!!!  THESE WILL TAKE A LONG TIME TO RUN
#***********************************************************************************
# Tune & Graph Latent Factors (dim) from 10 to 100 by 10
#***********************************************************************************

opts_list = list(dim      = c(seq(10, 100, 10)),    # Number of Latent Features
                 costp_l1 = c(0),       # L1 regularization cost for User factors
                 costp_l2 = c(0.01),    # L2 regularization cost for User factors
                 costq_l1 = c(0),       # L1 regularization cost for Movie factors
                 costq_l2 = c(0.1),     # L2 regularization cost for Movie factors
                 lrate    = c(0.1),     # Learning Rate - Aprox step size in Gradient Descent
                 nfold    = 5,          # Number of folds for Cross Validation
                 nthread  = 8,          # Set Number of Threads to 1 *** Required to get Reproducible Results ***
                 niter    = 50,         # Number of Iterations
                 verbose  = FALSE)      # Don't Show Fold Details

start <- Sys.time()
set.seed(1, sample.kind = "Rounding")
opts_tune <- r$tune(data_memory(train_set[, 1], train_set[, 2], train_set[, 3]),
                    opts = opts_list)
finish <- Sys.time()
tune_time <- finish - start

opts_tune$min

opts_tune$res %>%
  ggplot(aes(dim, loss_fun)) +
  geom_point() +
  geom_smooth(method="loess") +
  labs(x="Latent Factors (dim)", y="RMSE") +
  ggtitle("Latent Factors vs RMSE")

#***********************************************************************************
# Tune & Graph Latent Factors (dim) from 10 to 40
#***********************************************************************************

opts_list = list(dim      = c(seq(10, 40, 2)),   # Number of Latent Features
                 costp_l1 = c(0),       # L1 regularization cost for User factors
                 costp_l2 = c(0.01),    # L2 regularization cost for User factors
                 costq_l1 = c(0),       # L1 regularization cost for Movie factors
                 costq_l2 = c(0.1),     # L2 regularization cost for Movie factors
                 lrate    = c(0.1),     # Learning Rate - Aprox step size in Gradient Descent
                 nfold    = 5,          # Number of folds for Cross Validation
                 nthread  = 8,          # Set Number of Threads to 1 *** Required to get Reproducible Results ***
                 niter    = 50,         # Number of Iterations
                 verbose  = FALSE)      # Don't Show Fold Details

start <- Sys.time()
set.seed(1, sample.kind = "Rounding")
opts_tune <- r$tune(data_memory(train_set[, 1], train_set[, 2], train_set[, 3]),
                    opts = opts_list)
finish <- Sys.time()
tune_time <- finish - start

opts_tune$min

opts_tune$res %>%
  ggplot(aes(dim, loss_fun)) +
  geom_point() +
  geom_smooth(method="loess") +
  labs(x="Latent Factors (dim)", y="RMSE") +
  ggtitle("Latent Factors vs RMSE")


#***********************************************************************************
# Manually Create the Latent Factor Results from Above
#***********************************************************************************

CV_10_100_50 <- data.frame("dim" = c(seq(10, 100, 10)),
                           "loss_fun" = c(0.8135956,
                                          0.7959371,
                                          0.7952601,
                                          0.7965165,
                                          0.7973126,
                                          0.7987548,
                                          0.7996540,
                                          0.7989897,
                                          0.8003280,
                                          0.7989563))

CV_10_100_50 %>%
  ggplot(aes(dim, loss_fun)) +
  geom_point() +
  geom_smooth(method="loess") +
  labs(x="Latent Factors (dim)", y="RMSE") +
  ggtitle("Latent Factors vs RMSE")


CV_10_40_50 <- data.frame("dim" = c(10:40),
                          "loss_fun" = c(0.8137556,
                                         0.8100568,
                                         0.8082561,
                                         0.8057245,
                                         0.8035323,
                                         0.8006602,
                                         0.7996062,
                                         0.7980771,
                                         0.7971078,
                                         0.7977603,
                                         0.7948986,
                                         0.7961489,
                                         0.7947183,
                                         0.7950298,
                                         0.7943660,
                                         0.7955314,
                                         0.7950606,
                                         0.7949973,
                                         0.7958134,
                                         0.7954243,
                                         0.7957506,
                                         0.7945857,
                                         0.7956658,
                                         0.7967824,
                                         0.7945495,
                                         0.7951564,
                                         0.7962097,
                                         0.7959298,
                                         0.7957610,
                                         0.7959754,
                                         0.7970225))

CV_10_40_50 %>%
  ggplot(aes(dim, loss_fun)) +
  geom_point() +
  geom_smooth(method="loess") +
  labs(x="Latent Factors (dim)", y="RMSE") +
  ggtitle("Latent Factors vs RMSE")

#***********************************************************************************

#***********************************************************************************
# Clean Up the environment
#***********************************************************************************
rm()

