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


# Define our evaluation function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


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

#***********************************************************************************
# NOTE: This model will return the RMSE of 0.7808364 and not the "Best" RMSE in
# the report.  The largest model takes almost an hour to train.  This one will take
# a few minutes.
#***********************************************************************************

#***********************************************************************************
# V1 - 30 x 50
#***********************************************************************************

opts_list = list(dim      = c(30),      # Number of Latent Features
                 costp_l1 = c(0),       # L1 regularization cost for User factors
                 costp_l2 = c(0.01),    # L2 regularization cost for User factors
                 costq_l1 = c(0),       # L1 regularization cost for Movie factors
                 costq_l2 = c(0.1),     # L2 regularization cost for Movie factors
                 lrate    = c(0.1),     # Learning Rate - Aprox step size in Gradient Descent
                 niter    = 50,         # Number of Iterations for Training (Not used in Tuning)
                 nfolds   = 5,          # Number of Folds for CV in Tuning
                 verbose  = FALSE,      # Don't Show Progress
                 nthread  = 1)          #!!! Can be set to higher values for Tuning, but MUST be set to 1
                                        #    for Training or the results are not reproducible

#***********************************************************************************
# TRAIN the Model over 50 Iterations
#***********************************************************************************
start <- Sys.time()

set.seed(1, sample.kind = "Rounding")
r$train(data_memory(train_set[, 1], train_set[, 2], train_set[, 3]),
        opts = c(opts_list,
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
# Write out the submission.csv file
#***********************************************************************************

write.csv(pred, "submission.csv", row.names = FALSE)

#***********************************************************************************
# Read in the "rubric.csv" file and confirm the RMSE and predict against it
# The rubric.csv file is the same as the "validation" Dataframe with only the userId, movieId, and rating
#***********************************************************************************

rubric <- read.csv("rubric.csv")
rubric <- rubric %>% as.matrix()

pred <- r$predict(data_memory(rubric[, 1], rubric[, 2], rubric[, 3]), out_memory())

rmse <- RMSE(rubric[,3],pred)

RECO_rmse_results2 <- tibble(Method="V1 - 30 x 50 - Read.CSV",
                            RMSE = rmse,
                            Train_Time = round(as.numeric(train_time, units="mins"), 2))
RECO_rmse_results2 %>% knitr::kable()

