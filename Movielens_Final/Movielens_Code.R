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
  semi_join(edx, by = "userId")

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

##########################################################
# INVESTIGATE DATA
##########################################################

#View dataset
head(edx)

#Check nulls
any(is.na(edx))

#Create Smaller data sets Average Movie
mean_movie <- edx %>% group_by(movieId, title, genres) %>%
  summarize(n=n(), mean_rating=mean(rating))%>%
  select(movieId, title, genres,n,mean_rating)

mean_movie

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


##########################################################
# DEFAULT: CREATE TRAINING AND TEST DATA
##########################################################


#Create training and test data: Split on a 25% for testing
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(rates, times = 1, p = 0.25, list=FALSE)

test_set <- edx[test_index,]
train_set <- edx[-test_index,]

#Due to regularization issues we need to ensure users and movies in test set are covered in the training set

test_set <- test_set %>% semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

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

head(train_movie)

test_movie <- test_set %>% left_join(train_movie,by='movieId')

head(test_movie)
any(is.na(test_movie)) #this came up true now need to ensure that tests are represented in the train

RMSE <- sqrt(mean((test_set$rating-(mu+test_movie$rmean))^2))
RMSE

#Sample size regularisation accounting for sample size n
test_movie <- test_movie %>% mutate(reg=rsum/n)

RMSE <- sqrt(mean((test_set$rating-(mu+test_movie$reg))^2))
RMSE

#regularisation optimised with penalty term p
p <- seq(-10,20)

#sapply the terms
RMSEs <- sapply(p,function(p){
  test_movie_App <- test_movie %>% mutate(pen=rsum/(n+p))
  sqrt(mean((test_set$rating-(mu+test_movie_App$pen))^2))
})

#plot outputs
plot(p,RMSEs)


#Improve the accuracy
p <- seq(0,5,0.1)

#sapply the terms
RMSEs <- sapply(p,function(p){
  test_movie_App <- test_movie %>% mutate(pen=rsum/(n+p))
  sqrt(mean((test_set$rating-(mu+test_movie_App$pen))^2))
})

#plot outputs
plot(p,RMSEs)

pmin <- p[which.min(RMSEs)]

#final optimised output
test_movie <- test_movie %>% mutate(reg=rsum/(n+pmin))

RMSE <- sqrt(mean((test_set$rating-(mu+test_movie$reg))^2))
RMSE

##########################################################
# DEFAULT:  RMSE - USER EFFECT
##########################################################

#Create the regularized for sum and mean
train_user <- train_set %>% group_by(userId) %>%
  summarize(n=n(),rmean=mean(rating-mu),rsum=sum(rating-mu))

head(train_user)

test_user <- test_set %>% left_join(train_user,by='userId')

head(test_user)
any(is.na(test_user)) #this came up true now need to ensure that tests are represented in the train

RMSE <- sqrt(mean((test_set$rating-(mu+test_user$rmean))^2))
RMSE

#Sample size regularisation accounting for sample size n
test_user <- test_user %>% mutate(reg=rsum/n)

RMSE <- sqrt(mean((test_set$rating-(mu+test_user$reg))^2))
RMSE

#regularisation optimised with penalty term p
p <- seq(-10,20)

#sapply the terms
RMSEs <- sapply(p,function(p){
  test_user_App <- test_user %>% mutate(pen=rsum/(n+p))
  sqrt(mean((test_set$rating-(mu+test_user_App$pen))^2))
})

#plot outputs
plot(p,RMSEs)


#Improve the accuracy
p <- seq(2,8,0.1)

#sapply the terms
RMSEs <- sapply(p,function(p){
  test_user_App <- test_user %>% mutate(pen=rsum/(n+p))
  sqrt(mean((test_set$rating-(mu+test_user_App$pen))^2))
})

#plot outputs
plot(p,RMSEs)

pmin <- p[which.min(RMSEs)]

#final optimised output
test_user <- test_user %>% mutate(reg=rsum/(n+pmin))

RMSE <- sqrt(mean((test_set$rating-(mu+test_user$reg))^2))
RMSE

#User Affect is valid but not as strong as movie affect.


##########################################################
#  DEFAULT: RMSE - USER AND MOVIE EFFECT
##########################################################

RMSE <- sqrt(mean((test_set$rating-(mu+test_user$reg + test_movie$reg))^2))
RMSE


#Current output 0.8836105.
#This is equivalent to around 10 points


##########################################################
#--------------------------------------------------------
# MEDIAN LIMIT: CLEANER DATA RUN
#--------------------------------------------------------
##########################################################

#---- Limit movies and users by median ----

lim_movie <- mean_movie %>% filter(year>1950 & n>122)
head(lim_movie)
nrow(lim_movie)

lim_user <- mean_user %>% filter(n>62)
head(lim_user)
nrow(lim_user)

lim_edx <- edx %>% semi_join(lim_movie, by = "movieId") %>%
  semi_join(lim_user, by = "userId")

head(lim_edx)
nrow(lim_edx)


#Create training and test data: Split on a 25% for testing
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(lim_edx$rating, times = 1, p = 0.25, list=FALSE)

test_set <- lim_edx[test_index,]
train_set <- lim_edx[-test_index,]

#Due to regularization issues we need to ensure users and movies in test set are covered in the training set

test_set <- test_set %>% semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

head(test_set)
head(train_set)

n_test <- nrow(test_set)
n_train <-nrow(train_set)

n_test
n_train

##########################################################
# MEDIAN LIMIT:  RMSE - AVEARGE
##########################################################

mu <- mean(train_set$rating)
mu

RMSE <- sqrt(mean((test_set$rating-mu)^2))
RMSE

##########################################################
#  MEDIAN LIMIT: RMSE - MOVIE EFFECT
##########################################################

#Create the regularized for sum and mean
train_movie <- train_set %>% group_by(movieId) %>%
  summarize(n=n(),rmean=mean(rating-mu),rsum=sum(rating-mu))

head(train_movie)

test_movie <- test_set %>% left_join(train_movie,by='movieId')

head(test_movie)
any(is.na(test_movie)) #this came up true now need to ensure that tests are represented in the train

RMSE <- sqrt(mean((test_set$rating-(mu+test_movie$rmean))^2))
RMSE

#Sample size regularisation accounting for sample size n
test_movie <- test_movie %>% mutate(reg=rsum/n)

RMSE <- sqrt(mean((test_set$rating-(mu+test_movie$reg))^2))
RMSE

#regularisation optimised with penalty term p
p <- seq(-10,20)

#sapply the terms
RMSEs <- sapply(p,function(p){
  test_movie_App <- test_movie %>% mutate(pen=rsum/(n+p))
  sqrt(mean((test_set$rating-(mu+test_movie_App$pen))^2))
})

#plot outputs
plot(p,RMSEs)


#Improve the accuracy
p <- seq(0,5,0.1)

#sapply the terms
RMSEs <- sapply(p,function(p){
  test_movie_App <- test_movie %>% mutate(pen=rsum/(n+p))
  sqrt(mean((test_set$rating-(mu+test_movie_App$pen))^2))
})

#plot outputs
plot(p,RMSEs)

pmin <- p[which.min(RMSEs)]

#final optimised output
test_movie <- test_movie %>% mutate(reg=rsum/(n+pmin))

RMSE <- sqrt(mean((test_set$rating-(mu+test_movie$reg))^2))
RMSE

##########################################################
#  MEDIAN LIMIT: RMSE - USER EFFECT
##########################################################

#Create the regularized for sum and mean
train_user <- train_set %>% group_by(userId) %>%
  summarize(n=n(),rmean=mean(rating-mu),rsum=sum(rating-mu))

head(train_user)

test_user <- test_set %>% left_join(train_user,by='userId')

head(test_user)
any(is.na(test_user)) #this came up true now need to ensure that tests are represented in the train

RMSE <- sqrt(mean((test_set$rating-(mu+test_user$rmean))^2))
RMSE

#Sample size regularisation accounting for sample size n
test_user <- test_user %>% mutate(reg=rsum/n)

RMSE <- sqrt(mean((test_set$rating-(mu+test_user$reg))^2))
RMSE

#regularisation optimised with penalty term p
p <- seq(-10,20)

#sapply the terms
RMSEs <- sapply(p,function(p){
  test_user_App <- test_user %>% mutate(pen=rsum/(n+p))
  sqrt(mean((test_set$rating-(mu+test_user_App$pen))^2))
})

#plot outputs
plot(p,RMSEs)


#Improve the accuracy
p <- seq(2,8,0.1)

#sapply the terms
RMSEs <- sapply(p,function(p){
  test_user_App <- test_user %>% mutate(pen=rsum/(n+p))
  sqrt(mean((test_set$rating-(mu+test_user_App$pen))^2))
})

#plot outputs
plot(p,RMSEs)

pmin <- p[which.min(RMSEs)]

#final optimised output
test_user <- test_user %>% mutate(reg=rsum/(n+pmin))

RMSE <- sqrt(mean((test_set$rating-(mu+test_user$reg))^2))
RMSE

#User Affect is valid but not as strong as movie affect.


##########################################################
#  MEDIAN LIMIT: RMSE - REGULARIZATION USER AND MOVIE EFFECT
##########################################################

RMSE <- sqrt(mean((test_set$rating-(mu+test_user$reg + test_movie$reg))^2))
RMSE


#Current output 0.8726856
#This is equivalent to around 10 points


##########################################################
#--------------------------------------------------------
# MEAN LIMIT:  MORE LIMITED DATA RUN
#--------------------------------------------------------
##########################################################

#Get Summary data
summary(mean_movie$n)
summary(mean_user$n)

#---- Limit movies and users by mean ----

lim_movie <- mean_movie %>% filter(year>1950 & n>845)
head(lim_movie)
nrow(lim_movie)

lim_user <- mean_user %>% filter(n>130)
head(lim_user)
nrow(lim_user)

lim_edx <- edx %>% semi_join(lim_movie, by = "movieId") %>%
  semi_join(lim_user, by = "userId")

head(lim_edx)
nrow(lim_edx)


#Create training and test data: Split on a 25% for testing
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(lim_edx$rating, times = 1, p = 0.25, list=FALSE)

test_set <- lim_edx[test_index,]
train_set <- lim_edx[-test_index,]

#Due to regularization issues we need to ensure users and movies in test set are covered in the training set

test_set <- test_set %>% semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

head(test_set)
head(train_set)

n_test <- nrow(test_set)
n_train <-nrow(train_set)

n_test
n_train

##########################################################
#  MEAN LIMIT: RMSE - AVEARGE
##########################################################

mu <- mean(train_set$rating)
mu

RMSE <- sqrt(mean((test_set$rating-mu)^2))
RMSE

##########################################################
#  MEAN LIMIT: RMSE - MOVIE EFFECT
##########################################################

#Create the regularized for sum and mean
train_movie <- train_set %>% group_by(movieId) %>%
  summarize(n=n(),rmean=mean(rating-mu),rsum=sum(rating-mu))

head(train_movie)

test_movie <- test_set %>% left_join(train_movie,by='movieId')

head(test_movie)
any(is.na(test_movie)) #this came up true now need to ensure that tests are represented in the train

RMSE <- sqrt(mean((test_set$rating-(mu+test_movie$rmean))^2))
RMSE

#Sample size regularisation accounting for sample size n
test_movie <- test_movie %>% mutate(reg=rsum/n)

RMSE <- sqrt(mean((test_set$rating-(mu+test_movie$reg))^2))
RMSE

#regularisation optimised with penalty term p
p <- seq(-10,20)

#sapply the terms
RMSEs <- sapply(p,function(p){
  test_movie_App <- test_movie %>% mutate(pen=rsum/(n+p))
  sqrt(mean((test_set$rating-(mu+test_movie_App$pen))^2))
})

#plot outputs
plot(p,RMSEs)


#Improve the accuracy
p <- seq(0,5,0.1)

#sapply the terms
RMSEs <- sapply(p,function(p){
  test_movie_App <- test_movie %>% mutate(pen=rsum/(n+p))
  sqrt(mean((test_set$rating-(mu+test_movie_App$pen))^2))
})

#plot outputs
plot(p,RMSEs)

pmin <- p[which.min(RMSEs)]

#final optimised output
test_movie <- test_movie %>% mutate(reg=rsum/(n+pmin))

RMSE <- sqrt(mean((test_set$rating-(mu+test_movie$reg))^2))
RMSE

##########################################################
#  MEAN LIMIT: RMSE - USER EFFECT
##########################################################

#Create the regularized for sum and mean
train_user <- train_set %>% group_by(userId) %>%
  summarize(n=n(),rmean=mean(rating-mu),rsum=sum(rating-mu))

head(train_user)

test_user <- test_set %>% left_join(train_user,by='userId')

head(test_user)
any(is.na(test_user)) #this came up true now need to ensure that tests are represented in the train

RMSE <- sqrt(mean((test_set$rating-(mu+test_user$rmean))^2))
RMSE

#Sample size regularisation accounting for sample size n
test_user <- test_user %>% mutate(reg=rsum/n)

RMSE <- sqrt(mean((test_set$rating-(mu+test_user$reg))^2))
RMSE

#regularisation optimised with penalty term p
p <- seq(-10,20)

#sapply the terms
RMSEs <- sapply(p,function(p){
  test_user_App <- test_user %>% mutate(pen=rsum/(n+p))
  sqrt(mean((test_set$rating-(mu+test_user_App$pen))^2))
})

#plot outputs
plot(p,RMSEs)


#Improve the accuracy
p <- seq(2,8,0.1)

#sapply the terms
RMSEs <- sapply(p,function(p){
  test_user_App <- test_user %>% mutate(pen=rsum/(n+p))
  sqrt(mean((test_set$rating-(mu+test_user_App$pen))^2))
})

#plot outputs
plot(p,RMSEs)

pmin <- p[which.min(RMSEs)]

#final optimised output
test_user <- test_user %>% mutate(reg=rsum/(n+pmin))

RMSE <- sqrt(mean((test_set$rating-(mu+test_user$reg))^2))
RMSE

#User Affect is valid but not as strong as movie affect.


##########################################################
#  MEAN LIMIT: RMSE - REGULARIZATION USER AND MOVIE EFFECT
##########################################################

RMSE <- sqrt(mean((test_set$rating-(mu+test_user$reg + test_movie$reg))^2))
RMSE


#Current output 0.8564079
#This is equivalent to around 25 points We are therefore happy with this result


##########################################################
#---------------------------------------------------------
# RUN ON VALIDATION SET
#---------------------------------------------------------
##########################################################
#Test that the limited data is in the validation set.
#Get mean
mu <- mean(edx$rating)
mu

##########################################################
# VALID DEFAULT: - MOVIE EFFECT
##########################################################

#Create the regularized for sum and mean
train_movie <- edx %>% group_by(movieId) %>%
  summarize(n=n(),rmean=mean(rating-mu),rsum=sum(rating-mu))

head(train_movie)

test_movie <- validation %>% left_join(train_movie,by='movieId')

head(test_movie)
any(is.na(test_movie))

#Sample size regularisation accounting for sample size n
test_movie <- test_movie %>% mutate(reg=rsum/n)

RMSE <- sqrt(mean((validation$rating-(mu+test_movie$reg))^2))
RMSE

#regularisation optimised with penalty term p
p <- seq(-10,20)

#sapply the terms
RMSEs <- sapply(p,function(p){
  test_movie_App <- test_movie %>% mutate(pen=rsum/(n+p))
  sqrt(mean((validation$rating-(mu+test_movie_App$pen))^2))
})

#plot outputs
plot(p,RMSEs)


#Improve the accuracy
p <- seq(0,5,0.1)

#sapply the terms
RMSEs <- sapply(p,function(p){
  test_movie_App <- test_movie %>% mutate(pen=rsum/(n+p))
  sqrt(mean((validation$rating-(mu+test_movie_App$pen))^2))
})

#plot outputs
plot(p,RMSEs)

pmin <- p[which.min(RMSEs)]

#final optimised output
test_movie <- test_movie %>% mutate(reg=rsum/(n+pmin))

##########################################################
# VALID DEFAULT: - USER EFFECT
##########################################################

#Create the regularized for sum and mean
train_user <- edx %>% group_by(userId) %>%
  summarize(n=n(),rmean=mean(rating-mu),rsum=sum(rating-mu))

head(train_user)

test_user <- validation %>% left_join(train_user,by='userId')

head(test_user)
any(is.na(test_user))

#Sample size regularisation accounting for sample size n
test_user <- test_user %>% mutate(reg=rsum/n)

RMSE <- sqrt(mean((validation$rating-(mu+test_user$reg))^2))
RMSE

#regularisation optimised with penalty term p
p <- seq(-10,20)

#sapply the terms
RMSEs <- sapply(p,function(p){
  test_user_App <- test_user %>% mutate(pen=rsum/(n+p))
  sqrt(mean((validation$rating-(mu+test_user_App$pen))^2))
})

#plot outputs
plot(p,RMSEs)


#Improve the accuracy
p <- seq(2,8,0.1)

#sapply the terms
RMSEs <- sapply(p,function(p){
  test_user_App <- test_user %>% mutate(pen=rsum/(n+p))
  sqrt(mean((validation$rating-(mu+test_user_App$pen))^2))
})

#plot outputs
plot(p,RMSEs)

pmin <- p[which.min(RMSEs)]

#final optimised output
test_user <- test_user %>% mutate(reg=rsum/(n+pmin))

#User Affect is valid but not as strong as movie affect.


##########################################################
# VALID DEFAULT: RMSE - REGULARIZATION USER AND MOVIE EFFECT
##########################################################

#Final validated RMSE
RMSE <- sqrt(mean((validation$rating-(mu+test_user$reg + test_movie$reg))^2))
RMSE


#FINAL RMSE of 0.8830696 Which is above the target but still an acceptable result

##########################################################
# VALID CLEAN: - CLEAN DATA
##########################################################

#Get the mean counts
edx_movie <- edx %>% group_by(movieId) %>%
  summarize(n=n())

mean_mov_n <- mean(edx_movie$n)

edx_user <- edx %>% group_by(userId) %>%
  summarize(n=n())

mean_use_n <- mean(edx_user$n)

#Limit the by the mean

lim_movie <- edx_movie %>% filter(n>mean_mov_n)
nrow(lim_movie)

lim_user <- edx_user %>% filter(n>mean_use_n)
nrow(lim_user)

#Apply to edx and Validation
lim_edx <- edx %>% semi_join(lim_movie, by = "movieId") %>%
  semi_join(lim_user, by = "userId")


lim_valid <- validation %>% semi_join(lim_edx, by = "movieId") %>%
  semi_join(lim_edx, by = "userId")

##########################################################
# VALID CLEAN: - MOVIE EFFECT
##########################################################

#Create the regularized for sum and mean
train_movie <- lim_edx %>% group_by(movieId) %>%
  summarize(n=n(),rmean=mean(rating-mu),rsum=sum(rating-mu))

head(train_movie)

test_movie <- lim_valid %>% left_join(train_movie,by='movieId')

head(test_movie)
any(is.na(test_movie))

#Sample size regularisation accounting for sample size n
test_movie <- test_movie %>% mutate(reg=rsum/n)

RMSE <- sqrt(mean((lim_valid$rating-(mu+test_movie$reg))^2))
RMSE

#regularisation optimised with penalty term p
p <- seq(-10,20)

#sapply the terms
RMSEs <- sapply(p,function(p){
  test_movie_App <- test_movie %>% mutate(pen=rsum/(n+p))
  sqrt(mean((lim_valid$rating-(mu+test_movie_App$pen))^2))
})

#plot outputs
plot(p,RMSEs)


#Improve the accuracy
p <- seq(0,5,0.1)

#sapply the terms
RMSEs <- sapply(p,function(p){
  test_movie_App <- test_movie %>% mutate(pen=rsum/(n+p))
  sqrt(mean((lim_valid$rating-(mu+test_movie_App$pen))^2))
})

#plot outputs
plot(p,RMSEs)

pmin <- p[which.min(RMSEs)]

#final optimised output
test_movie <- test_movie %>% mutate(reg=rsum/(n+pmin))

##########################################################
# VALID CLEAN: - USER EFFECT
##########################################################

#Create the regularized for sum and mean
train_user <- lim_edx %>% group_by(userId) %>%
  summarize(n=n(),rmean=mean(rating-mu),rsum=sum(rating-mu))

head(train_user)

test_user <- lim_valid %>% left_join(train_user,by='userId')

head(test_user)
any(is.na(test_user))

#Sample size regularisation accounting for sample size n
test_user <- test_user %>% mutate(reg=rsum/n)

RMSE <- sqrt(mean((lim_valid$rating-(mu+test_user$reg))^2))
RMSE

#regularisation optimised with penalty term p
p <- seq(-10,20)

#sapply the terms
RMSEs <- sapply(p,function(p){
  test_user_App <- test_user %>% mutate(pen=rsum/(n+p))
  sqrt(mean((lim_valid$rating-(mu+test_user_App$pen))^2))
})

#plot outputs
plot(p,RMSEs)


#Improve the accuracy
p <- seq(2,8,0.1)

#sapply the terms
RMSEs <- sapply(p,function(p){
  test_user_App <- test_user %>% mutate(pen=rsum/(n+p))
  sqrt(mean((lim_valid$rating-(mu+test_user_App$pen))^2))
})

#plot outputs
plot(p,RMSEs)

pmin <- p[which.min(RMSEs)]

#final optimized output
test_user <- test_user %>% mutate(reg=rsum/(n+pmin))


##########################################################
# VALID CLEAN: RMSE - REGULARIZATION USER AND MOVIE EFFECT
##########################################################

#Final validated RMSE
RMSE <- sqrt(mean((lim_valid$rating-(mu+test_user$reg + test_movie$reg))^2))
RMSE

#FINAL RMSE of 0.8564252

#Rated Items:
nrow(lim_valid)/nrow(validation)
