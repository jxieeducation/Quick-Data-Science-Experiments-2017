## This is shell code to estimate the ATE 

## load data
gg <- read.csv("http://hdl.handle.net/10079/ghx3frr")
df <- read.csv("~/Desktop/personal_git/experiments-causality/data/GerberGreenBook_Chapter3_Donations.csv")
head(df)
tail(df)
df

mean(df$Y[1:10]) - mean(df$Y[11,20])


## create a function to randomize 
# 1. how many treatment and control assignents 
#    do you want?  10 each?
random <- function() { 
  sample(c(rep(0,10),rep(1,10)))
}

random()

df$Y * rep(0, 20)

## create a function to calculate the ATE 
estAte <- function(data=df, assignment=random()){
  mean(df$Y * assignment) - mean(df$Y * (1-assignment))
}

## calculate the averages in treatment and control 
estAte()
estAte(assignment=df$Z)

## use estAte() with the data you have and your randomizer 
## figure out what 100 (1000?) distributions *coulld* have 
## been

results = replicate(n = 100, estAte())
results
hist(results)

# p-val
sum(results > estAte(assignment=df$Z)) / 100 # 1-tail
sum(abs(results) > estAte(assignment=df$Z)) / 100 # 2-tail
t.test(df[df$Z == 0,]$Y, df[df$Z == 1, ]$Y) # t-test
