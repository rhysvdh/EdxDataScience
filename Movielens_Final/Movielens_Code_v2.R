#**********************************************************************
#*
#* MOVIELENS CODE
#* Rhys van den Handel
#* October 2020
#* 
#**********************************************************************

#Data Preparation - GIVEN
##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                           title = as.character(title),
#                                          genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movies
movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId") %>%
  semi_join(edx, by = "genres")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

##########################################################
# PACKAGE LOADING
##########################################################

library(dplyr)
library(tidyverse)
library(caret)
library(data.table)
library(gsubfn)
library(stringr)

################################################################################
#---------------------------------------------------------
# DATA ANALYSIS
#---------------------------------------------------------
################################################################################

################################################################################
# EDX DATA
################################################################################

#View dataset
head(edx)

#Check nulls
any(is.na(edx))

################################################################################
# MOVIES
################################################################################

#Create Smaller data sets Average Movie
mean_movie <- edx %>% group_by(movieId, title, genres) %>%
  summarize(n=n(), mean_rating=mean(rating))%>%
  select(movieId, title, genres,n,mean_rating)

#Count number of movies
nrow(mean_movie)

#movie stats
summary(mean_movie$n)

summary(mean_movie$mean_rating)

#10 most rated movies
mean_movie %>% arrange(desc(n)) %>%
  top_n(10,n)

#10 best rated movies
mean_movie %>% arrange(desc(mean_rating)) %>%
  top_n(10,mean_rating)

#10 best rated movies > 122 ratings (median)
mean_movie %>% filter(n>122) %>%
  arrange(desc(mean_rating)) %>%
  top_n(10,mean_rating)

#pull year from title
as.numeric(str_sub(mean_movie$title,-5,-2)) #works to pull year in numeric format

#test on meanmovie
mean_movie %>% mutate(year=as.numeric(str_sub(title,-5,-2)))
mean_movie <- mean_movie%>% mutate(year=as.numeric(str_sub(title,-5,-2)))

#Apply to edx
edx_yr<-edx %>% mutate(year=as.numeric(str_sub(title,-5,-2)))

#plot mean ratings by year and number of movies
mean_movie %>% group_by(year) %>%
  summarize(yn=n(),yr_mean_rate=mean(mean_rating))%>%
  ggplot(aes(x=year)) +
  geom_point(aes(y=yn/100), color="blue") +
  geom_point(aes(y=yr_mean_rate), color="red")+
  scale_y_continuous(name = "Mean Rating",sec.axis = sec_axis(~.*100, name="Number Movies"))


################################################################################
# USERS
################################################################################

#Users
mean_user <- edx_yr %>% group_by(userId) %>%
  summarize(n=n(), mean_rating=mean(rating))%>%
  select(userId,n,mean_rating)

mean_user

#Count number of users
nrow(mean_user)

#user stats
summary(mean_user$n)
summary(mean_user$mean_rating)


hist(mean_user$n) #there are some outliers with most being <1000
hist(mean_user$n,xlim=c(1,1000),breaks=100)

#10 users with most ratings
mean_user %>% arrange(desc(n)) %>%
  top_n(10,n)

#10 users with best ratings
mean_user %>% arrange(desc(mean_rating)) %>%
  top_n(10,mean_rating)

#10 users with best ratings > 62 ratings (median)
mean_user %>% filter(n>62) %>%
  arrange(desc(mean_rating)) %>%
  top_n(10,mean_rating)

#10 users with worst ratings > 62 ratings (median)
mean_user %>% filter(n>62) %>%
  arrange(mean_rating)

################################################################################
# Genres
################################################################################

#Look at interactions between different genres and the rating.
boxplot(edx$rating~edx$genres)

#There are too many interactions to ascertain conclusive evidence

#Create Smaller data sets Average Movie
mean_genre <- edx %>% group_by(genres) %>%
  summarize(n=n(), mean_rating=mean(rating))%>%
  select(genres,n,mean_rating)

head(mean_genre)

#Count number of movies
nrow(mean_genre)

#movie stats
summary(mean_genre$n)
summary(mean_genre$mean_rating)

#10 most rated genres
mean_genre %>% arrange(desc(n)) %>%
  top_n(10,n)

#10 best rated genres
mean_genre %>% arrange(desc(mean_rating)) %>%
  top_n(10,mean_rating)

################################################################################
#---------------------------------------------------------
# PREDICTION MODEL
#---------------------------------------------------------
################################################################################

##########################################################
# DEFAULT: CREATE TRAINING AND TEST DATA
##########################################################


#Create training and test data: Split on a 25% for testing
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(edx$rating, times = 1, p = 0.1, list=FALSE)

test_set <- edx[test_index,]
train_set <- edx[-test_index,]

#Due to regularization issues we need to ensure users and movies in test set are covered in the training set

test_set <- test_set %>% semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId") %>%
  semi_join(train_set, by = "genres")

head(test_set)
head(train_set)

n_test <- nrow(test_set)
n_train <-nrow(train_set)

n_test
n_train

##########################################################
# DEFAULT:  RMSE - AVEARGE
##########################################################

mu <- mean(train_set$rating)
mu

#Running an RMSE on setting the prediction as the mean of the dataset.
RMSE <- sqrt(mean((test_set$rating-mu)^2))
RMSE

##########################################################
#  DEFAULT: RMSE - MOVIE EFFECT
##########################################################

#Create the regularized for sum and mean
train_movie <- train_set %>% group_by(movieId) %>%
  summarize(n=n(),rmean=mean(rating-mu),rsum=sum(rating-mu))


#Apply to test and training set
test_movie <- test_set %>% left_join(train_movie,by='movieId')

any(is.na(test_movie))
any(is.na(train_movie))

#Sample size regularisation accounting for sample size n
test_movie <- test_movie %>% mutate(reg=rsum/n)

RMSE_mov_nopen <- sqrt(mean((test_set$rating-(mu+test_movie$reg))^2))
RMSE_mov_nopen

#regularisation optimised with penalty term p
p <- seq(-10,20,0.5)

#sapply the terms
RMSEs <- sapply(p,function(p){
  train_movie <- train_movie %>% mutate(pen=rsum/(n+p))
  tune_movie <- test_set %>% left_join(train_movie,by='movieId')
  sqrt(mean((test_set$rating-(mu+tune_movie$pen))^2))
})

#plot outputs
plot(p,RMSEs)

#Find optimal penalty term
pmin_mov <- p[which.min(RMSEs)]
pmin_mov

#final optimised output
test_movie <- test_movie %>% mutate(reg=rsum/(n+pmin_mov))

RMSE_mov <- sqrt(mean((test_set$rating-(mu+test_movie$reg))^2))
RMSE_mov

##########################################################
# DEFAULT:  RMSE - USER EFFECT
##########################################################

#Create the regularized for sum and mean
train_user <- train_set %>% group_by(userId) %>%
  summarize(n=n(),rmean=mean(rating-mu),rsum=sum(rating-mu))

test_user <- test_set %>% left_join(train_user,by='userId')

any(is.na(test_user)) 
any(is.na(train_user)) 


#Sample size regularization accounting for sample size n
test_user <- test_user %>% mutate(reg=rsum/n)

RMSE_use_nopen <- sqrt(mean((test_set$rating-(mu+test_user$reg))^2))
RMSE_use_nopen

#regularization optimized with penalty term p
p <- seq(-10,20)

#sapply the terms
RMSEs <- sapply(p,function(p){
  train_user <- train_user %>% mutate(pen=rsum/(n+p))
  tune_user <- test_set %>% left_join(train_user,by='userId')
  sqrt(mean((test_set$rating-(mu+tune_user$pen))^2))
})

#plot outputs
plot(p,RMSEs)


#Improve the accuracy
p <- seq(2,8,0.1)

#sapply the terms
RMSEs <- sapply(p,function(p){
  train_user <- train_user %>% mutate(pen=rsum/(n+p))
  tune_user <- test_set %>% left_join(train_user,by='userId')
  sqrt(mean((test_set$rating-(mu+tune_user$pen))^2))
})

#plot outputs
plot(p,RMSEs)

pmin_use <- p[which.min(RMSEs)]

#final optimized output
test_user <- test_user %>% mutate(reg=rsum/(n+pmin_use))

RMSE_use <- sqrt(mean((test_set$rating-(mu+test_user$reg))^2))
RMSE_use

#User Affect is valid but not as strong as movie affect.

##########################################################
#  DEFAULT: RMSE - Genre EFFECT
##########################################################

#Create the regularized for sum and mean
train_genre <- train_set %>% group_by(genres) %>%
  summarize(n=n(),rmean=mean(rating-mu),rsum=sum(rating-mu))


#Apply to test and training set
test_genre <- test_set %>% left_join(train_genre,by='genres')

any(is.na(test_genre))
any(is.na(train_genre))

#Sample size regularisation accounting for sample size n
test_genre <- test_genre %>% mutate(reg=rsum/n)

RMSE_gen_nopen <- sqrt(mean((test_set$rating-(mu+test_genre$reg))^2))
RMSE_gen_nopen

#regularisation optimised with penalty term p
p <- seq(-10,20,0.5)

#sapply the terms
RMSEs <- sapply(p,function(p){
  train_genre <- train_genre %>% mutate(pen=rsum/(n+p))
  tune_genre <- test_set %>% left_join(train_genre,by='genres')
  sqrt(mean((test_set$rating-(mu+tune_genre$pen))^2))
})

#plot outputs
plot(p,RMSEs)

#Find optimal peanalty term
pmin_gen <- p[which.min(RMSEs)]
pmin_gen

#final optimised output
test_genre <- test_genre %>% mutate(reg=rsum/(n+pmin_gen))

RMSE_gen <- sqrt(mean((test_set$rating-(mu+test_genre$reg))^2))
RMSE_gen

##########################################################
#  DEFAULT: RMSE - USER AND MOVIE EFFECT
##########################################################

RMSE <- sqrt(mean((test_set$rating-(mu+test_user$reg + test_movie$reg + test_genre$reg))^2))
RMSE

#Improve the accuracy
p <- seq(50,150,0.5)

#sapply the terms
RMSEs <- sapply(p,function(p){
  train_user <- train_user %>% mutate(pen_u=rsum/(n+p))
  train_movie <- train_movie %>% mutate(pen_m=rsum/(n+p))
  train_genre <- train_genre %>% mutate(pen_g=rsum/(n+p))
  tune_comb <- test_set %>% 
    left_join(train_user,by='userId') %>% 
    left_join(train_movie,by='movieId')%>% 
    left_join(train_genre,by='genres')
  sqrt(mean((test_set$rating-(mu+tune_comb$pen_u+tune_comb$pen_m+tune_comb$pen_g))^2))
})

#plot outputs
plot(p,RMSEs)

pmin_comb <- p[which.min(RMSEs)]

#final optimized output
test_user <- test_user %>% mutate(reg=rsum/(n+pmin_comb))
test_movie <- test_movie %>% mutate(reg=rsum/(n+pmin_comb))

RMSE_comb <- sqrt(mean((test_set$rating-(mu+test_user$reg+test_movie$reg))^2))
RMSE_comb


#Current output 0.8773574
#This is equivalent to around 10 points

###############################################################################
# RESULTS
###############################################################################

#Build results table
results_table <- data.frame(Method='Default Mean', RMSE = RMSE_mu)
results_table <- bind_rows(results_table,data.frame(Method='Default Movie Mean', RMSE = RMSE_mov_nopen))
results_table <- bind_rows(results_table,data.frame(Method='Default Movie tuned', RMSE = RMSE_mov))
results_table <- bind_rows(results_table,data.frame(Method='Default User Mean', RMSE = RMSE_use_nopen))
results_table <- bind_rows(results_table,data.frame(Method='Default User tuned', RMSE = RMSE_use))
results_table <- bind_rows(results_table,data.frame(Method='Default Genre Mean', RMSE = RMSE_gen_nopen))
results_table <- bind_rows(results_table,data.frame(Method='Default Genre tuned', RMSE = RMSE_gen))
results_table <- bind_rows(results_table,data.frame(Method='Default Combined Model', RMSE = RMSE_comb))
results_table %>% knitr::kable()

summary_table <- data.frame(Method='Default Combined Model', RMSE = RMSE_comb)

##########################################################
#--------------------------------------------------------
# MEDIAN LIMIT: CLEANER DATA RUN
#--------------------------------------------------------
##########################################################

#---- Limit movies users and genres by median ----

#Get the mean counts
edx_movie <- edx %>% group_by(movieId) %>%
  summarize(n=n())

edx_user <- edx %>% group_by(userId) %>%
  summarize(n=n())

edx_genre <- edx %>% group_by(genres) %>%
  summarize(n=n())

median_mov_n <- median(edx_movie$n)
median_use_n <- median(edx_user$n)
median_gen_n <- median(edx_genre$n)

#Limit the by the mean

lim_movie <- edx_movie %>% filter(n>median_mov_n)
lim_user <- edx_user %>% filter(n>median_use_n)
lim_genre <- edx_genre %>% filter(n>median_gen_n)

#Apply to edx
lim_edx <- edx %>% semi_join(lim_movie, by = "movieId") %>%
  semi_join(lim_user, by = "userId") %>%
  semi_join(lim_genre, by = "genres")


head(lim_edx)
nrow(lim_edx)


#Create training and test data: Split on a 10% for testing
set.seed(1, sample.kind = "Rounding")
lim_test_index <- createDataPartition(lim_edx$rating, times = 1, p = 0.1, list=FALSE)

lim_test_set <- lim_edx[lim_test_index,]
lim_train_set <- lim_edx[-lim_test_index,]

#Due to regularization issues we need to ensure users and movies in test set are covered in the training set

lim_test_set <- lim_test_set %>% semi_join(lim_train_set, by = "movieId") %>%
  semi_join(lim_train_set, by = "userId") %>%
  semi_join(lim_train_set, by = "genres")

n_test <- nrow(lim_test_set)
n_train <-nrow(lim_train_set)

n_test
n_train

##########################################################
# MEDIAN LIMIT:  RMSE - AVEARGE
##########################################################

mu <- mean(lim_train_set$rating)
mu

RMSE_mu <- sqrt(mean((lim_test_set$rating-mu)^2))
RMSE_mu

##########################################################
#  MEDIAN LIMIT: RMSE - MOVIE EFFECT
##########################################################

#Create the regularized for sum and mean
train_movie <- lim_train_set %>% group_by(movieId) %>%
  summarize(n=n(),rmean=mean(rating-mu),rsum=sum(rating-mu))


#Apply to test and training set
test_movie <- lim_test_set %>% left_join(train_movie,by='movieId')

any(is.na(test_movie))
any(is.na(train_movie))

#Sample size regularisation accounting for sample size n
test_movie <- test_movie %>% mutate(reg=rsum/n)

RMSE_mov_nopen <- sqrt(mean((lim_test_set$rating-(mu+test_movie$reg))^2))
RMSE_mov_nopen

#regularisation optimised with penalty term p
p <- seq(-10,20,0.5)

#sapply the terms
RMSEs <- sapply(p,function(p){
  train_movie <- train_movie %>% mutate(pen=rsum/(n+p))
  tune_movie <- lim_test_set %>% left_join(train_movie,by='movieId')
  sqrt(mean((lim_test_set$rating-(mu+tune_movie$pen))^2))
})

#plot outputs
plot(p,RMSEs)

#Find optimal peanalty term
pmin_mov <- p[which.min(RMSEs)]
pmin_mov

#final optimised output
test_movie <- test_movie %>% mutate(reg=rsum/(n+pmin_mov))

RMSE_mov <- sqrt(mean((lim_test_set$rating-(mu+test_movie$reg))^2))
RMSE_mov

##########################################################
#  MEDIAN LIMIT: RMSE - USER EFFECT
##########################################################

#Create the regularized for sum and mean
train_user <- lim_train_set %>% group_by(userId) %>%
  summarize(n=n(),rmean=mean(rating-mu),rsum=sum(rating-mu))

test_user <- lim_test_set %>% left_join(train_user,by='userId')

any(is.na(test_user)) 
any(is.na(train_user)) 


#Sample size regularization accounting for sample size n
test_user <- test_user %>% mutate(reg=rsum/n)

RMSE_use_nopen <- sqrt(mean((lim_test_set$rating-(mu+test_user$reg))^2))
RMSE_use_nopen

#regularization optimized with penalty term p
p <- seq(-10,20,0.5)

#sapply the terms
RMSEs <- sapply(p,function(p){
  train_user <- train_user %>% mutate(pen=rsum/(n+p))
  tune_user <- lim_test_set %>% left_join(train_user,by='userId')
  sqrt(mean((lim_test_set$rating-(mu+tune_user$pen))^2))
})

#plot outputs
plot(p,RMSEs)

pmin_use <- p[which.min(RMSEs)]

#final optimized output
test_user <- test_user %>% mutate(reg=rsum/(n+pmin_use))

RMSE_use <- sqrt(mean((lim_test_set$rating-(mu+test_user$reg))^2))
RMSE_use
#User effect is valid but not as strong as movie effect.

##########################################################
#  MEDIAN LIMIT:: RMSE - GENRE EFFECT
##########################################################

#Create the regularized for sum and mean
train_genre <- lim_train_set %>% group_by(genres) %>%
  summarize(n=n(),rmean=mean(rating-mu),rsum=sum(rating-mu))


#Apply to test and training set
test_genre <- lim_test_set %>% left_join(train_genre,by='genres')

any(is.na(test_genre))
any(is.na(train_genre))

#Sample size regularisation accounting for sample size n
test_genre <- test_genre %>% mutate(reg=rsum/n)

RMSE_gen_nopen <- sqrt(mean((lim_test_set$rating-(mu+test_genre$reg))^2))
RMSE_gen_nopen

#regularisation optimised with penalty term p
p <- seq(-10,20,0.5)

#sapply the terms
RMSEs <- sapply(p,function(p){
  train_genre <- train_genre %>% mutate(pen=rsum/(n+p))
  tune_genre <- lim_test_set %>% left_join(train_genre,by='genres')
  sqrt(mean((lim_test_set$rating-(mu+tune_genre$pen))^2))
})

#plot outputs
plot(p,RMSEs)

#Find optimal peanalty term
pmin_gen <- p[which.min(RMSEs)]
pmin_gen

#final optimised output
test_genre <- test_genre %>% mutate(reg=rsum/(n+pmin_gen))

RMSE_gen <- sqrt(mean((lim_test_set$rating-(mu+test_genre$reg))^2))
RMSE_gen

##########################################################
#  MEDIAN LIMIT: RMSE - REGULARIZATION USER AND MOVIE EFFECT
##########################################################

RMSE <- sqrt(mean((lim_test_set$rating-(mu+test_user$reg + test_movie$reg + test_genre$reg))^2))
RMSE

#Improve the accuracy
p <- seq(75,150,1)

#sapply the terms
RMSEs <- sapply(p,function(p){
  train_user <- train_user %>% mutate(pen_u=rsum/(n+p))
  train_movie <- train_movie %>% mutate(pen_m=rsum/(n+p))
  train_genre <- train_genre %>% mutate(pen_g=rsum/(n+p))
  tune_comb <- lim_test_set %>% 
    left_join(train_user,by='userId') %>% 
    left_join(train_movie,by='movieId')%>% 
    left_join(train_genre,by='genres')
  sqrt(mean((lim_test_set$rating-(mu+tune_comb$pen_u+tune_comb$pen_m+tune_comb$pen_g))^2))
})

#plot outputs
plot(p,RMSEs)

pmin_comb_lim <- p[which.min(RMSEs)]

#final optimized output
test_user <- test_user %>% mutate(reg=rsum/(n+pmin_comb_lim))
test_movie <- test_movie %>% mutate(reg=rsum/(n+pmin_comb_lim))

RMSE_comb <- sqrt(mean((lim_test_set$rating-(mu+test_user$reg+test_movie$reg))^2))
RMSE_comb

#Current output 0.8726321
#This is equivalent to around 10 points

###############################################################################
# RESULTS
###############################################################################

#Append to results table
results_table <- bind_rows(results_table,data.frame(Method='Limited Mean', RMSE = RMSE_mu))
results_table <- bind_rows(results_table,data.frame(Method='Limited Movie Mean', RMSE = RMSE_mov_nopen))
results_table <- bind_rows(results_table,data.frame(Method='Limited Movie tuned', RMSE = RMSE_mov))
results_table <- bind_rows(results_table,data.frame(Method='Limited User Mean', RMSE = RMSE_use_nopen))
results_table <- bind_rows(results_table,data.frame(Method='Limited User tuned', RMSE = RMSE_use))
results_table <- bind_rows(results_table,data.frame(Method='Limited Genre Mean', RMSE = RMSE_gen_nopen))
results_table <- bind_rows(results_table,data.frame(Method='Limited Genre tuned', RMSE = RMSE_gen))
results_table <- bind_rows(results_table,data.frame(Method='Limited Combined Model', RMSE = RMSE_comb))
results_table

summary_table <- bind_rows(summary_table,data.frame(Method='Limited Combined Model', RMSE = RMSE_comb))

################################################################################
#---------------------------------------------------------
# RUN ON VALIDATION SET
#---------------------------------------------------------
################################################################################

#Get mean
mu <- mean(edx$rating)
mu

##########################################################
# VALID DEFAULT: - MOVIE EFFECT
##########################################################

#Create the regularized for sum and mean
train_movie <- edx %>% group_by(movieId) %>%
  summarize(n=n(),rmean=mean(rating-mu),rsum=sum(rating-mu))

val_movie <- validation %>% left_join(train_movie,by='movieId')
any(is.na(val_movie))

#Sample size regularization accounting for sample size n
val_movie <- val_movie %>% mutate(reg=rsum/(n+pmin_comb))

##########################################################
# VALID DEFAULT: - USER EFFECT
##########################################################

#Create the regularized for sum and mean
train_user <- edx %>% group_by(userId) %>%
  summarize(n=n(),rmean=mean(rating-mu),rsum=sum(rating-mu))

val_user <- validation %>% left_join(train_user,by='userId')
any(is.na(val_user))

#Sample size regularization accounting for sample size n
val_user <- val_user %>% mutate(reg=rsum/(n+pmin_comb))

##########################################################
# VALID DEFAULT: - GENRE EFFECT
##########################################################

#Create the regularized for sum and mean
train_genre <- edx %>% group_by(genres) %>%
  summarize(n=n(),rmean=mean(rating-mu),rsum=sum(rating-mu))

val_genre <- validation %>% left_join(train_genre,by='genres')
any(is.na(val_genre))

#Sample size regularization accounting for sample size n
val_genre <- val_genre %>% mutate(reg=rsum/(n+pmin_comb))

##########################################################
# VALID DEFAULT: RMSE - REGULARIZATION USER AND MOVIE EFFECT
##########################################################

#Final validated RMSE
RMSE_Val <- sqrt(mean((validation$rating-(mu+val_user$reg+val_movie$reg +val_genre$reg))^2))
RMSE_Val

###############################################################################
# RESULTS
###############################################################################

#Append to results table
results_table <- bind_rows(results_table,data.frame(Method='Default Validation', RMSE = RMSE_Val))

summary_table <- bind_rows(summary_table,data.frame(Method='Default Validation', RMSE = RMSE_Val))


##########################################################
# VALID CLEAN: - CLEAN DATA
##########################################################

#Get the mean counts
edx_movie <- edx %>% group_by(movieId) %>%
  summarize(n=n())

edx_user <- edx %>% group_by(userId) %>%
  summarize(n=n())

edx_genre <- edx %>% group_by(genres) %>%
  summarize(n=n())

median_mov_n <- median(edx_movie$n)
median_use_n <- median(edx_user$n)
median_gen_n <- median(edx_genre$n)

#Limit the by the mean

lim_movie <- edx_movie %>% filter(n>median_mov_n)
lim_user <- edx_user %>% filter(n>median_use_n)
lim_genre <- edx_genre %>% filter(n>median_gen_n)

#Apply to edx
lim_edx <- edx %>% semi_join(lim_movie, by = "movieId") %>%
  semi_join(lim_user, by = "userId") %>%
  semi_join(lim_genre, by = "genres")


lim_valid <- validation %>% semi_join(lim_edx, by = "movieId") %>%
  semi_join(lim_edx, by = "userId") %>%
  semi_join(lim_edx, by = "genres")

any(is.na(lim_valid))


##########################################################
# VALID CLEAN: - MOVIE EFFECT
##########################################################

#Get mean
mu <- mean(lim_edx$rating)

#Create the regularized for sum and mean
train_movie <- lim_edx %>% group_by(movieId) %>%
  summarize(n=n(),rmean=mean(rating-mu),rsum=sum(rating-mu))


val_movie <- lim_valid %>% left_join(train_movie,by='movieId')
any(is.na(val_movie))

#Sample size regularization accounting for sample size n
val_movie <- val_movie %>% mutate(reg=rsum/(n+pmin_comb_lim))

##########################################################
# VALID CLEAN: - USER EFFECT
##########################################################

#Create the regularized for sum and mean
train_user <- lim_edx %>% group_by(userId) %>%
  summarize(n=n(),rmean=mean(rating-mu),rsum=sum(rating-mu))

val_user <- lim_valid %>% left_join(train_user,by='userId')
any(is.na(val_user))

#Sample size regularization accounting for sample size n
val_user <- val_user %>% mutate(reg=rsum/(n+pmin_comb_lim))

##########################################################
# VALID CLEAN: - GENRE EFFECT
##########################################################

#Create the regularized for sum and mean
train_genre <- lim_edx %>% group_by(genres) %>%
  summarize(n=n(),rmean=mean(rating-mu),rsum=sum(rating-mu))

val_genre <- lim_valid %>% left_join(train_genre,by='genres')
any(is.na(val_genre))

#Sample size regularization accounting for sample size n
val_genre <- val_genre %>% mutate(reg=rsum/(n+pmin_comb_lim))

##########################################################
# VALID CLEAN: RMSE - REGULARIZATION USER AND MOVIE EFFECT
##########################################################

#Final validated RMSE
RMSE_Val <- sqrt(mean((lim_valid$rating-(mu+val_user$reg+val_movie$reg+val_genre$reg))^2))
RMSE_Val


#Rated Items:
nrow(lim_valid)/nrow(validation)

#Unfortunately this method simply does not evaluate over 40% of the data . 
#This is poor therefore it is recommended that the RMSE of 0.8830696 be taken as it evaluated 100% of the data.

###############################################################################
# RESULTS
###############################################################################

#Append to results table
results_table <- bind_rows(results_table,data.frame(Method='Limited Validation', RMSE = RMSE_Val))
summary_table <- bind_rows(summary_table,data.frame(Method='Limited Validation', RMSE = RMSE_Val))
results_table 
summary_table 
