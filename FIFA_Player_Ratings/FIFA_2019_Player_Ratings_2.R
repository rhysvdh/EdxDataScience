################################################################################
# AUTHOR:   Rhys Jan van den Handel
# DATE:     November 2020
# Name:     FIFA 2019 Player Ratings
################################################################################

################################################################################
# ------------------------------------------------------------------------------
# PROJECT PREPARATION
# ------------------------------------------------------------------------------
################################################################################

##########################################################
# PACKAGE LOADING
##########################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(gsubfn)) install.packages("gsubfn", repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(gsubfn)
library(stringr)
library(dplyr)

##########################################################
# LOAD DATA
##########################################################

#Download Data from internet
temp <- tempfile()

url <- "https://www.kaggle.com/karangadiya/fifa19/download/archive.zip"
download.file(url, temp)
unzip(temp, "data.csv")
data<-read.csv("data.csv", header = TRUE)

unlink(temp)


#Read the data in from project
readcsv <- read.csv(".\\data.csv", header = TRUE)

#Replace data with readcsv for using project dataset
PlayerData <- data.frame(data)

#view the dataset
head(PlayerData)
any(is.na(PlayerData)) #True: therefore there are NA's
nrow(PlayerData) # 18207

#Remove the NA's
PlayerData <- PlayerData %>% drop_na()
nrow(PlayerData) # 18147 Therefore, only 60 rows dropped


##########################################################
# LOAD INTO WORKING AND VALIDATION SET
##########################################################

# Validation set will be 10% of the dataset
set.seed(1, sample.kind="Rounding")

test_index <- createDataPartition(y = PlayerData$Overall, times = 1, p = 0.1, list = FALSE)
players <- PlayerData[-test_index,]
validation <- PlayerData[test_index,]

################################################################################
# ------------------------------------------------------------------------------
# DATA ANALYSIS AND PREPARATION
# ------------------------------------------------------------------------------
################################################################################

##########################################################
# DATA EXPLORATION
##########################################################

#Checking the players data set
names(players)

head(players)
any(is.na(players))

#Distribution of player ratings
hist(players$Overall)

#Summary statistics of rating
summary(players$Overall)

#--- AGE SUMMARY -----------------------------------------

#Distribution of player ages
hist(players$Age)

ages <- players %>% group_by(Age) %>%
  summarize(n=n(),mean_rate=mean(Overall))

#Plot of number of players and average rating by age
ages %>% ggplot(aes(x=Age)) +
  geom_point(aes(y=n/20), color="blue") +
  geom_point(aes(y=mean_rate), color="red")+
  scale_y_continuous(name = "Mean Rating",sec.axis = sec_axis(~.*20, name="Number Players")) +
  ggtitle("Average Ratings and Number of Players by Age")

#--- Height and Weight -----------------------------------------

hwcols <-  c("Name","Age","Overall","Height","Weight","Body.Type")
hw <- players %>% select(hwcols)
temp <- hw %>% mutate(Weight=as.numeric(str_sub(Weight,1,-4)),ft=as.numeric(str_sub(Height,1,1)),inc=as.numeric(str_sub(Height,3,-1)))
hw <- temp %>% mutate(Height=((inc*0.0254)+(ft*0.3048))) %>% select(hwcols)

#Distribution of height and weight
hist(hw$Height)
hist(hw$Weight)

#Height and Weight vs Rating

hw_rating <- hw %>% group_by(Overall) %>%
  summarize(n=n(),height=mean(Height),weight=mean(Weight))

hw_rating %>%  ggplot(aes(x=Overall)) +
  geom_point(aes(y=weight/100), color="blue") +
  geom_point(aes(y=height), color="red")+
  scale_y_continuous(name = "Mean Height (Red)",sec.axis = sec_axis(~.*100, name="Mean weight (Blue)")) +
  ggtitle("Average Height and Weight of Players by Rating")

#So there is is a correlation in Weight but not in height

boxplot(hw$Overall~hw$Body.Type)

#Body Type does not give anything useful. The player names as body types are not useful and will make using it to predict ratings difficult.

#--- Nationality -----------------------------------------

nation <- players %>% group_by(Nationality) %>%
  summarize(n=n(),rating=mean(Overall)) 

summary(nation$n)

#Nations with most players
nation %>% arrange(desc(n)) %>%
  top_n(10,n)

#Nations with least players
nation %>% arrange((n)) %>%
  top_n(10,-n)

#Nations with best players
nation %>% filter(n>12) %>%
  arrange(desc(rating)) %>%
  top_n(10,rating)

#Nations with worst players
nation %>% filter(n>12) %>%
  arrange((rating)) %>%
  top_n(10,-rating)

#Nationality has an impact but due to the inconsistent number of players of each nation it would be difficult to use.

#--- Clubs -----------------------------------------

clubs <- players %>% group_by(Club) %>%
  summarize(n=n(),rating=mean(Overall)) 

summary(clubs$n)

#Clubs with most players
clubs %>% arrange(desc(n)) %>%
  top_n(10,n)

#Clubs with least players
clubs %>% arrange((n)) %>%
  top_n(10,-n)

#Clubs with best players
clubs %>% arrange(desc(rating)) %>%
  top_n(10,rating)

#Clubs with worst players
clubs %>% arrange((rating)) %>%
  top_n(10,-rating)

#Clubs are definitely a good option for training

#--- Jersey.Number -----------------------------------------

Jersey <- players %>% group_by(Jersey.Number) %>%
  summarize(n=n(),rating=mean(Overall)) 

summary(Jersey$n)

#Clubs with most players
Jersey %>% arrange(desc(n)) %>%
  top_n(10,n)

#Clubs with least players
Jersey %>% arrange((n)) %>%
  top_n(10,-n)

#Clubs with best players
Jersey %>% arrange(desc(rating)) %>%
  top_n(10,rating)

#Clubs with worst players
Jersey %>% arrange((rating)) %>%
  top_n(10,-rating)

#Jersey Numbers are definitely a good option for training. However, due to non linearity regularization should be used

#--- Position and Skills -----------------------------------------

pscols <-  c("Name","Age","Overall","Special","International.Reputation","Weak.Foot","Skill.Moves","Work.Rate","Position")

ps <- players %>% select(pscols)

#Work.Rate is not an easy item to quantify

boxplot(ps$Overall~ps$Position)
boxplot(ps$Overall~ps$International.Reputation)
boxplot(ps$Overall~ps$Weak.Foot)
boxplot(ps$Overall~ps$Skill.Moves)


#Position is not helpful but International Reputation, Weak foot and Skill moves Will be useful. 

#--- MONETARY VALUES -----------------------------------------

moncols <- c("Name","Age","Overall","Value","Wage","Release.Clause")

money <- players %>% select(moncols)
temp <- money %>% mutate(value_unit=(str_sub(Value,-1,-1)),value_euro=as.numeric(str_sub(Value,4,-2)),wage_unit=(str_sub(Wage,-1,-1)),wage_euro=as.numeric(str_sub(Wage,4,-2)),release_unit=(str_sub(Release.Clause,-1,-1)),release_euro=as.numeric(str_sub(Release.Clause,4,-2)))
money <- temp %>% mutate(Value=ifelse(value_unit=="K",value_euro*1000,value_euro*1000000),Wage=ifelse(wage_unit=="K",wage_euro*1000,wage_euro*1000000),Release.Clause=ifelse(release_unit=="K",release_euro*1000,release_euro*1000000)) %>%
  select(moncols)
money[is.na(money)] <- 0


money_rating <- money %>% group_by(Overall) %>%
  summarize(n=n(),value=mean(Value),wage=mean(Wage),release=mean(Release.Clause))

plot(money_rating$Overall,money_rating$wage)
plot(money_rating$Overall,money_rating$value)
plot(money_rating$Overall,money_rating$release)

#All Monetary values have a large impact at higher overall ratings.


################################################################################
# ------------------------------------------------------------------------------
# PREDICTIVE MODEL
# ------------------------------------------------------------------------------
################################################################################

##########################################################
# DATA PREPARATION
##########################################################


#--- Columns to be used ---

header <- c("Name","Age","Overall","Weight","Value","Wage","Release.Clause","International.Reputation","Weak.Foot","Skill.Moves","Club","Jersey.Number")

players <- players %>% select(header)
validation <- validation %>% select(header)
#--- players data set ---

#Fix monetary formatting


temp <- players %>% mutate(value_unit=(str_sub(Value,-1,-1)),value_euro=as.numeric(str_sub(Value,4,-2)),wage_unit=(str_sub(Wage,-1,-1)),wage_euro=as.numeric(str_sub(Wage,4,-2)),release_unit=(str_sub(Release.Clause,-1,-1)),release_euro=as.numeric(str_sub(Release.Clause,4,-2)))
players <- temp %>% mutate(Value=ifelse(value_unit=="K",value_euro*1000,value_euro*1000000),Wage=ifelse(wage_unit=="K",wage_euro*1000,wage_euro*1000000),Release.Clause=ifelse(release_unit=="K",release_euro*1000,release_euro*1000000)) %>%
  select(header)

temp <- validation %>% mutate(value_unit=(str_sub(Value,-1,-1)),value_euro=as.numeric(str_sub(Value,4,-2)),wage_unit=(str_sub(Wage,-1,-1)),wage_euro=as.numeric(str_sub(Wage,4,-2)),release_unit=(str_sub(Release.Clause,-1,-1)),release_euro=as.numeric(str_sub(Release.Clause,4,-2)))
validation <- temp %>% mutate(Value=ifelse(value_unit=="K",value_euro*1000,value_euro*1000000),Wage=ifelse(wage_unit=="K",wage_euro*1000,wage_euro*1000000),Release.Clause=ifelse(release_unit=="K",release_euro*1000,release_euro*1000000)) %>%
  select(header)

#Fix weight format

temp <- players %>% mutate(Weight=as.numeric(str_sub(Weight,1,-4)))
players <- temp %>% select(header)

temp <- validation %>% mutate(Weight=as.numeric(str_sub(Weight,1,-4)))
validation <- temp %>% select(header)

#Ensure N/As are set as 0
players[is.na(players)] <- 0
validation[is.na(validation)] <- 0

#--- View Data set ---

#players
head(players)
nrow(players) #Should be 16331
any(is.na(players))

# validation
head(validation)
nrow(validation) #Should be 1816
any(is.na(validation))


#--- Clear Memory ---
rm(ages,clubs,hw,hw_rating,money,money_rating,nation,PlayerData,ps,readcsv,temp,Jersey)

##########################################################
# LOAD INTO TEST AND TRAIN DATA
##########################################################

# Test set will be 10% of the dataset
set.seed(1, sample.kind="Rounding")

test_index <- createDataPartition(y = players$Overall, times = 1, p = 0.1, list = FALSE)
train_set <- players[-test_index,]
test_set <- players[test_index,]



##########################################################
# BASIC STATISTICS
##########################################################

summary(train_set$Overall)
mu <- mean(train_set$Overall)
med <- median(train_set$Overall)

##########################################################
# MEAN MODEL
##########################################################

#Running an RMSE on mean
RMSE_mu <- sqrt(mean((test_set$Overall-mu)^2))
RMSE_mu

#Absolute Error
AbsError_mu <- mean(abs(mu-test_set$Overall))
AbsError_mu

##########################################################
# CARET GLM MODEL
##########################################################

#--- AGE AND WEIGHT PREDICTION------------------------------#

#run a GLM
fit_aw <- train(Overall~Age+Weight,data=train_set, method = "glm")
fit_aw

train_pred_aw <-predict(fit_aw,train_set)
pred_aw <-predict(fit_aw,test_set)

RMSE_aw <- sqrt(mean((test_set$Overall-pred_aw)^2))
RMSE_aw

#Absolute Error
AbsError_aw <- mean(abs(pred_aw-test_set$Overall))
AbsError_aw

#--- SKILL, WEAK FOOT AND REPUTATION PREDICTION ----------#

#run a GLM
fit_swr <- train(Overall~Skill.Moves+Weak.Foot+International.Reputation,data=train_set, method = "glm")
fit_swr

train_pred_swr <-predict(fit_swr,train_set)
pred_swr <-predict(fit_swr,test_set)

RMSE_swr <- sqrt(mean((test_set$Overall-pred_swr)^2))
RMSE_swr

#Absolute Error
AbsError_swr <- mean(abs(pred_swr-test_set$Overall))
AbsError_swr

#--- MONETARY PREDICTION ------------------------------#

#run a GLM
fit_mon <- train(Overall~Value + Wage + Release.Clause,data=train_set, method = "glm")
fit_mon

train_pred_mon <-predict(fit_mon,train_set)
pred_mon <-predict(fit_mon,test_set)

#Running an RMSE to view the error
RMSE_mon_glm <- sqrt(mean((test_set$Overall-pred_mon)^2))
RMSE_mon_glm

#Absolute Error
AbsError_mon <- mean(abs(pred_mon-test_set$Overall))
AbsError_mon

##########################################################
# REGULARISATION
##########################################################


#--- Club -----------------------------------------------#

#Create the regularized for sum and mean
club <- train_set %>% group_by(Club) %>%
  summarize(n=n(),rsum=sum(Overall-mu))

t_club <- test_set %>% left_join(club,by='Club')
train_club <- train_set %>% left_join(club,by='Club')

#Sample size regularization accounting for sample size n
t_club <- t_club %>% mutate(b_club=rsum/n)

RMSE_club_nopen <- sqrt(mean((test_set$Overall-(mu+t_club$b_club))^2))
RMSE_club_nopen

#regularization optimized with penalty term p
p <- seq(-10,20)

#sapply the terms
RMSEs <- sapply(p,function(p){
  club <- club %>% mutate(b_club=rsum/(n+p))
  train_club <- train_set %>% left_join(club,by='Club')
  sqrt(mean((train_set$Overall-(mu+train_club$b_club))^2))
})

#plot outputs
plot(p,RMSEs)

#Improve accuracy
p <- seq(-5,5,0.2)

#sapply the terms
RMSEs <- sapply(p,function(p){
  club <- club %>% mutate(b_club=rsum/(n+p))
  train_club <- train_set %>% left_join(club,by='Club')
  sqrt(mean((train_set$Overall-(mu+train_club$b_club))^2))
})

#plot outputs
plot(p,RMSEs)

p_club <- p[which.min(RMSEs)]
p_club

#final optimized output
t_club <- t_club %>% mutate(b_club=rsum/(n+p_club))

RMSE_club <- sqrt(mean((test_set$Overall-(mu+t_club$b_club))^2))
RMSE_club

#Absolute Error
AbsError_club <- mean(abs((mu+t_club$b_club)-test_set$Overall))
AbsError_club


#--- Jersey Number -----------------------------------------------#

#Create the regularized for sum and mean
Jersey <- train_set %>% group_by(Jersey.Number) %>%
  summarize(n=n(),rsum=sum(Overall-mu))

t_jersey <- test_set %>% left_join(Jersey,by='Jersey.Number')
train_jn <- train_set %>% left_join(Jersey,by='Jersey.Number')

#Sample size regularization accounting for sample size n
t_jersey <- t_jersey %>% mutate(b_jersey=rsum/n)

RMSE_jn_nopen <- sqrt(mean((test_set$Overall-(mu+t_jersey$b_jersey))^2))
RMSE_jn_nopen

#regularization optimized with penalty term p
p <- seq(-10,20)

#sapply the terms
RMSEs <- sapply(p,function(p){
  Jersey <- Jersey %>% mutate(b_jersey=rsum/(n+p))
  train_jn <- train_set %>% left_join(Jersey,by='Jersey.Number')
  sqrt(mean((train_set$Overall-(mu+train_jn$b_jersey))^2))
})

#plot outputs
plot(p,RMSEs)

#Improve accuracy
p <- seq(-1,5,0.2)

#sapply the terms
RMSEs <- sapply(p,function(p){
  Jersey <- Jersey %>% mutate(b_jersey=rsum/(n+p))
  train_jn <- train_set %>% left_join(Jersey,by='Jersey.Number')
  sqrt(mean((train_set$Overall-(mu+train_jn$b_jersey))^2))
})

#plot outputs
plot(p,RMSEs)

p_jersey <- p[which.min(RMSEs)]
p_jersey

#final optimized output
t_jersey <- t_jersey %>% mutate(b_jersey=rsum/(n+p_jersey))

RMSE_jn <- sqrt(mean((test_set$Overall-(mu+t_jersey$b_jersey))^2))
RMSE_jn

#Absolute Error
AbsError_jn <- mean(abs((mu+t_jersey$b_jersey)-test_set$Overall))
AbsError_jn


###############################################################################
# COMBINED REG
###############################################################################

#Create b_sum to hold all previous prediction values for tuning
b_sum = train_pred_mon

#Combine monetary and age weight values (Note these already contain mu)
train_b_sum = (train_pred_mon + train_pred_aw + train_pred_swr)/3
b_sum = (pred_mon + pred_aw + pred_swr)/3

#Using a more powerful computer to run all
fit_glm <- train(Overall~Age+Weight+Skill.Moves+Weak.Foot+International.Reputation+Value + Wage + Release.Clause,data=train_set, method = "glm")

train_b_sum <-predict(fit_glm,train_set)
b_sum <-predict(fit_glm,test_set)

#Results of the glm combined
RMSE_glm <- sqrt(mean((test_set$Overall-b_sum)^2))
RMSE_glm

#Absolute Error
AbsError_glm <- mean(abs(b_sum-test_set$Overall))
AbsError_glm

#COMBINE CLUB INTO MODEL WITH TUNING

#regularization optimized with penalty term p
p <- seq(0,90)

#sapply the terms
RMSEs <- sapply(p,function(p){
  club <- club %>% mutate(b_club=rsum/(n+p))
  train_club <- train_set %>% left_join(club,by='Club')
  sqrt(mean((train_set$Overall-(train_b_sum+train_club$b_club))^2))
})

#plot outputs
plot(p,RMSEs)

p_club <- p[which.min(RMSEs)]

#final optimized output
t_club <- t_club %>% mutate(b_club=rsum/(n+p_club))
train_club <- train_club %>% mutate(b_club=rsum/(n+p_club))

b_sum=b_sum+t_club$b_club
train_b_sum=train_b_sum+train_club$b_club

#COMBINE Jersey Number INTO MODEL WITH TUNING

#regularization optimized with penalty term p
p <- seq(1000,2000)

#sapply the terms
RMSEs <- sapply(p,function(p){
  Jersey <- Jersey %>% mutate(b_jersey=rsum/(n+p))
  train_jn <- train_set %>% left_join(Jersey,by='Jersey.Number')
  sqrt(mean((train_set$Overall-(train_b_sum+train_jn$b_jersey))^2))
})

#As the sequence gets larger the accuracy improvement is less effective therefore will not be used.

#plot outputs
plot(p,RMSEs)

p_jersey <- p[which.min(RMSEs)]

#final optimized output
t_jersey <- t_jersey %>% mutate(b_jersey=rsum/(n+p_jersey))
train_jn <- train_jn %>% mutate(b_jersey=rsum/(n+p_jersey))

b_sum=b_sum+t_jersey$b_jersey
train_b_sum=train_b_sum+train_jn$b_jersey

RMSE_combined <- sqrt(mean((test_set$Overall-b_sum)^2))
RMSE_combined

#Combined Error
AbsError_combined <- mean(abs(b_sum-test_set$Overall))
AbsError_combined

#Accuracy of prediction
acc <- round(b_sum,0) == test_set$Overall
mean(acc)*100

#Distribution of error
Error_Combined <-b_sum-test_set$Overall
hist(Error_Combined)

###############################################################################
# RESULTS
###############################################################################

#Build results table
results_table <- data.frame(Method='Mean Prediction', RMSE = RMSE_mu ,Error = AbsError_mu)
results_table <- bind_rows(results_table,data.frame(Method='Physical Prediction', RMSE = RMSE_aw ,Error = AbsError_aw))
results_table <- bind_rows(results_table,data.frame(Method='Club Prediction', RMSE = RMSE_club ,Error = AbsError_club))
results_table <- bind_rows(results_table,data.frame(Method='Jersey Number Prediction', RMSE = RMSE_jn ,Error = AbsError_jn))
results_table <- bind_rows(results_table,data.frame(Method='Simple Attributes Prediction', RMSE = RMSE_swr ,Error = AbsError_swr))
results_table <- bind_rows(results_table,data.frame(Method='Monetary Prediction', RMSE = RMSE_mon_glm ,Error = AbsError_mon))
results_table <- bind_rows(results_table,data.frame(Method='Combined GLM Results', RMSE = RMSE_glm,Error = AbsError_glm))
results_table <- bind_rows(results_table,data.frame(Method='Combined Results', RMSE = RMSE_combined,Error = AbsError_combined))
results_table


################################################################################
# ------------------------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------------------------
################################################################################

#individual glm models
val_mon <-predict(fit_mon,validation)
val_aw <-predict(fit_aw,validation)
val_swr <-predict(fit_swr,validation)


#Club regularization
club <- train_set %>% group_by(Club) %>%
  summarize(n=n(),rsum=sum(Overall-mu))

val_club <- validation %>% left_join(club,by='Club')
val_club <- val_club %>% mutate(b_club=rsum/(n+p_club))

#Jersey Number regularization
Jersey <- train_set %>% group_by(Jersey.Number) %>%
  summarize(n=n(),rsum=sum(Overall-mu))

val_jn <- validation %>% left_join(Jersey,by='Jersey.Number')
val_jn <- val_club %>% mutate(b_jersey=rsum/(n+p_jersey))

#Combine results
val_b_sum <-  (val_mon + val_aw + val_swr)/3

#With a more powerful computer the full glm can be run 
val_glm <-predict(fit_glm,validation)
val_b_sum <- val_glm

val_b_sum <- val_b_sum+val_club$b_club+val_jn$b_jersey

#Final result
RMSE_Final <- sqrt(mean((validation$Overall-val_b_sum)^2))
RMSE_Final

#Final Error
AbsError_Final <- mean(abs(val_b_sum-validation$Overall))
AbsError_Final

#Accuracy of prediction
acc <- round(val_b_sum,0) == validation$Overall
mean(acc)*100

#Distribution of error
val_error <- val_b_sum-validation$Overall
hist(val_error)

#Save into Results table
results_table <- bind_rows(results_table,data.frame(Method='Validation Results', RMSE = RMSE_Final, Error = AbsError_Final))
results_table




