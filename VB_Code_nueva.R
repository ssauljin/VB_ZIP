library(dbplyr)
library(pscl)
library(matrixStats)

## True values of latent variables theta, lambda, and pi
set.seed(100)
eta0 <-  1
eta1 <-  -2
alpha0  <- -2.5
alpha1  <-  2
M <- 5000
X <- rnorm(6*M, 0, 1)
X_train     <- matrix(                             X, ncol=6)[,-6]
pi_true     <- matrix(1/(1+exp(-eta0 - X*eta1)) , ncol=6)
lambda_true <- matrix(     exp(  alpha0 + X* alpha1)  , ncol=6)

time_BBVI1  + 1.27
time_Bayes1 + 1.27

gamma <- 3.8

theta  <- rgamma(M, shape=gamma, scale=1/gamma)

Y      <- cbind((1-rbinom(M, size=1, pi_true[,1]))*rpois(M, lambda_true[,1]*theta),
                (1-rbinom(M, size=1, pi_true[,2]))*rpois(M, lambda_true[,2]*theta),
                (1-rbinom(M, size=1, pi_true[,3]))*rpois(M, lambda_true[,3]*theta),
                (1-rbinom(M, size=1, pi_true[,4]))*rpois(M, lambda_true[,4]*theta),
                (1-rbinom(M, size=1, pi_true[,5]))*rpois(M, lambda_true[,5]*theta))

Ytest <- (1-rbinom(M, size=1, pi_true[,6]))*rpois(100*M, lambda_true[,6]*theta)
mean(Ytest)
system.time(
est_model <- zeroinfl(as.vector(Y) ~ as.vector(X_train))
)
summary(est_model)

eta0_est<- coef(est_model)[3]
eta1_est<- coef(est_model)[4]
alpha0_est <- coef(est_model)[1]
alpha1_est <- coef(est_model)[2]

system.time(
est_modelt <- glm(as.vector(Y) ~ as.vector(X_train), family=poisson(link="log"))
)
summary(est_modelt)

alpha0_estt <- coef(est_modelt)[1]
alpha1_estt <- coef(est_modelt)[2]

# eta0_est<- eta0
# eta1_est<- eta1
# alpha0_est <- alpha0
# alpha1_est <- alpha1

## number of people = 2000

# eta0 <-  1
# eta1 <- -2
# alpha0  <- -2.5
# alpha1  <-  2

##### Define functions 
# Sample from current Variational distribution
q_sample <- function(ns, y, g_q, zero_prob, lambda_t){
   theta_s = rgamma(ns, g_q + sum(y), 
                    rate = g_q + sum((1-zero_prob)*lambda_t) )
   
   return(theta_s)
}

# Negative log prior
neg_log_gamma <- function(x, a, b){
   return(-dgamma(x, shape=a, rate=b, log = T))
}

neg_log_prior <- function(theta_s, g_0){
   temp = neg_log_gamma(theta_s, g_0,g_0)
   return(temp)
}

# Negative log q (variational distribution)
neg_log_q <- function(theta_s, g_q, y, zero_prob, lambda_t){
   temp = neg_log_gamma(theta_s, g_q + sum(y),
                        g_q + sum((1-zero_prob)*lambda_t))
   return(temp)
}

# Negative log-likelihood (ZIP)
neg_log_likel <- function(theta_s, 
                          y, zero_prob, lambda_t){
   poi = dpois(y, outer(lambda_t, theta_s))
   
   term1 = (y == 0)*zero_prob
   term2 = (1-zero_prob)*poi
   
   term3 = colSums(log(term1 + term2))
   
   return(-term3)
}

# Empirical negative ELBO
neg_elbo <- function(y, theta_s, g_q, zero_prob, lambda_t){
   temp = neg_log_likel(theta_s, y, zero_prob, lambda_t) + 
      neg_log_prior(theta_s, g_0) -
      neg_log_q(theta_s, g_q, y, zero_prob, lambda_t)
   return(temp)
}


# Inverse logit function
inv_logit <- function(x){
   return(exp(x)/(1+exp(x)))
}

## derivative functions: dq/dgamma

dev_eta <- function(y, theta_s, eta, zero_prob, lambda_t){
   # g_q = log(1+exp(eta))
   g_q = logSumExp(c(0,eta))
   
   term1 = log(g_q + sum((1-zero_prob)*lambda_t))
   term2 = g_q / (g_q + sum((1-zero_prob)*lambda_t))
   term3 = sum(y) / (g_q + sum((1-zero_prob)*lambda_t))
   term4 = digamma(g_q + sum(y))
   term5 = log(theta_s) - theta_s
   
   temp = term1 + term2 + term3 - term4
   
   grad_eta = inv_logit(eta)
   
   return((temp + term5)*grad_eta)
}

# dev_gamma <- function(y, theta_s, g_q, zero_prob, lambda_t){
#   
#   term1 = log(g_q + sum((1-zero_prob)*lambda_t))
#   term2 = g_q / (g_q + sum((1-zero_prob)*lambda_t))
#   term3 = sum(y) / (g_q + sum((1-zero_prob)*lambda_t))
#   term4 = digamma(g_q + sum(y))
#   term5 = log(theta_s) - theta_s
#   
#   temp = term1 + term2 + term3 - term4
#   
#   return(temp + term5)
# }

#####################################
# Universal gamma_q ################
#####################################

## True value of thetas (random effects)
# n = M
set.seed(1)
m = 250
m_samp = sample(M, size = m)

total_time = 5
theta_true = theta

## Generating observation
lambda_it  <- matrix(     exp(  alpha0_est + X* alpha1_est)  , ncol=6)[,-6]

pars = theta_true*lambda_it # parameters of Poisson
pars %>% dim

set.seed(1)
ys = Y

zero_prob_it <- matrix(1/(1+exp(-eta0_est - X*eta1_est)) , ncol=6)[,-6]
zero_prob_it %>% dim

## Finding the optimal g_q 
parameter_update_universal <- function(lr, eta){ # lr: learning rate
   NE   = 0
   deta = 0
   g_q = logSumExp(c(0,eta))
   
   for(i in m_samp){
      y = ys[i,]
      zero_prob = zero_prob_it[i,]
      lambda_t = lambda_it[i,]
      theta_s = q_sample(ns, y, g_q, zero_prob, lambda_t)
      
      NE = NE + neg_elbo(y, theta_s, g_q, zero_prob, lambda_t)
      deta = deta + dev_eta(y, theta_s, eta, zero_prob, lambda_t)
   }
   
   eta <- eta - lr*mean(deta*NE)
   
   return(list(eta, mean(NE), mean(deta)))
}

##########
set.seed(1)
g_0 = 3
# g_q = 0.5
eta = 0.0

etas = c()
n_ELBOs = c()
detas = c()

ns=2000
n_update = 500


ptm.init <- proc.time()
system.time(
   for (it in 1:n_update) {
      
      result = parameter_update_universal(1/(1000+it), eta)
      
      eta = result[[1]]
      n_ELBO = result[[2]]
      deta = result[[3]]
      
      etas[it] = eta
      n_ELBOs[it] = n_ELBO
      detas[it] = deta
      if(it %% 50 == 0){
         print(it)
      }
   }
)
time_BBVI1=proc.time() - ptm.init

# negative ELBO plot
plot(1:length(n_ELBOs), n_ELBOs, type = 'l')

g_qs = log(1+exp(etas))
# g_qs

mini = which.min(n_ELBOs) 
opt_g1 = g_qs[mini]

n_ELBOs[mini]
opt_g1

library(coda)
library(rjags)
library(runjags)

ptm.init <- proc.time()
# data list
dataList=list(N=Y[,-6], X=matrix(X,ncol=6)[,-6], M=M,  
              alpha1 =  alpha1_est,  alpha0=  alpha0_est,
              eta0= eta0_est, eta1= eta1_est,
              c0=.001, d0=0.001)

# model
modelString="model {

## simulate R, N ############################# 
for (i in 1:M){
    theta[i] ~ dgamma(gam1, gam2)
    for(t in 1:5){
        R[i,t] ~ dbern( 1/(1+exp( -eta0-eta1*X[i,t])) )
        mu_N[i,t] = exp( alpha0+alpha1*X[i,t])*theta[i]
        N[i,t] ~ dpois( mu_N[i,t]*(1-R[i,t]) + 0.00001 )
    }
}

## prior #####################################
invsigsq ~ dgamma(c0,d0)
gam1 = invsigsq; gam2 = invsigsq
}
"


nChains=3; nAdapt=5000; nUpdate=30000; nSamples=30000; nthin=5
ptm.init <- proc.time()
runJagsOut = run.jags(method="parallel", model=modelString, 
                      monitor=c('theta'),
                      data=dataList,
                      n.chains=nChains, adapt=nAdapt, burnin=nUpdate,
                      sample=ceiling(nSamples/nChains), thin=nthin,
                      summarise=FALSE, plots=FALSE)
time_Bayes1=proc.time() - ptm.init

codaSamples = as.mcmc.list(runJagsOut)

mcmcSamples=mcmc(codaSamples[[1]])
if( nChains > 1) for(ich in 2:nChains) mcmcSamples=rbind(mcmcSamples,mcmc(codaSamples[[ich]]))
para.samples1 =  as.matrix(mcmcSamples)
Bayes_weight1 <- colMeans(para.samples1)
rm(codaSamples, mcmcSamples)


X_test      <- matrix(                             X, ncol=6)[,6]
cred_weight <- (rowSums(Y)+opt_g1) / (rowSums((1-zero_prob_it)*lambda_it)+opt_g1)
cre2_weight <- (rowSums(Y)+opt_g1) / (rowSums(matrix( exp(alpha0_estt + X* alpha1_estt)  , ncol=6)[,-6])+opt_g1)

naive_prm1  <- matrix(  1/(1+exp(eta0_est + X*eta1_est))*exp(alpha0_est + X* alpha1_est)  , ncol=6)[,6]
 cred_prm1  <- naive_prm1 * cred_weight
 true_prm1  <- naive_prm1 * theta
Bayes_prm1  <- naive_prm1 * Bayes_weight1

naiv2_prm1  <- matrix( exp(alpha0_estt + X* alpha1_estt)  , ncol=6)[,6]
 cre2_prm1  <- naiv2_prm1*cre2_weight

naive_diff1 <- naive_prm1 - Ytest
 cred_diff1 <-  cred_prm1 - Ytest
naiv2_diff1 <- naiv2_prm1 - Ytest
 cre2_diff1 <-  cre2_prm1 - Ytest
 
 true_diff1 <-  true_prm1 - Ytest
Bayes_diff1 <- Bayes_prm1 - Ytest
 
naive_RMSE1 <- mean(naive_diff1^2)
 cred_RMSE1 <- mean( cred_diff1^2)
naiv2_RMSE1 <- mean(naiv2_diff1^2)
 cre2_RMSE1 <- mean( cre2_diff1^2)
 true_RMSE1 <- mean( true_diff1^2)
Bayes_RMSE1 <- mean(Bayes_diff1^2)
 
naive_MAE1 <- mean(abs(naive_diff1))
 cred_MAE1 <- mean(abs( cred_diff1))
naiv2_MAE1 <- mean(abs(naiv2_diff1))
 cre2_MAE1 <- mean(abs( cre2_diff1))
 true_MAE1 <- mean(abs( true_diff1))
Bayes_MAE1 <- mean(abs(Bayes_diff1))
 

c(naive_RMSE1, cred_RMSE1, true_RMSE1, naiv2_RMSE1, cre2_RMSE1)
c(naive_MAE1 , cred_MAE1 , true_MAE1 , naiv2_MAE1 , cre2_MAE1 )


round(c(naiv2_RMSE1, cre2_RMSE1, naive_RMSE1, cred_RMSE1, Bayes_RMSE1, true_RMSE1 ),4)
round(c(naiv2_MAE1, cre2_MAE1, naive_MAE1, cred_MAE1, Bayes_MAE1, true_MAE1 ),4)

rbind(time_BBVI1, time_Bayes1)

 ###################
# Optimal gamma_q
# 1.951 # M=100
# 2.001 # M=500
# 1.927 # M=1000
# 2.002 # M = 3000

####

# load rawdata from URL
# load(url("https://sites.google.com/a/wisc.edu/jed-frees/home/documents/data.RData"))
# load(url("https://sites.google.com/a/wisc.edu/jed-frees/home/documents/dataout.RData"))


load("data.RData")
load("dataout.RData")

train <- data[,c(1:2,4,9:14,21:25)] # Only IM claim is used
test <- dataout[,c(1:2,4,9:14,21:25)] # Only IM claim is used

rm(data, dataout)
head(train)

dd <- aggregate(Year ~ PolicyNum, data = train, FUN = length)
ddd <- subset(dd, Year==5)
colnames(ddd)[2] <- "YCount"

dddd <- merge(train, ddd, by="PolicyNum")
train <- subset(dddd, YCount == 5)
length(unique(train$PolicyNum))
train <- train[order(train$PolicyNum, train$Year),]
train$YCount <- NULL

time1 + 0.64
70.90 + 0.64
system.time(
data_model <- zeroinfl(FreqIM ~ TypeCity+TypeCounty+TypeMisc+TypeSchool
                      +TypeTown+CoverageIM+lnDeductIM+NoClaimCreditIM, data=train)
)

summary(data_model)

system.time(
dat2_model <- glm(FreqIM ~ TypeCity+TypeCounty+TypeMisc+TypeSchool
                       +TypeTown+CoverageIM+lnDeductIM+NoClaimCreditIM, data=train,
                  family=poisson(link="log"))
)
summary(dat2_model)

zitest(dat2_model, type = c("scoreZIP"))

 alpha_est <- coef(data_model)[1 :9 ]
eta_est <- coef(data_model)[10:18]

 alpha_estt <- coef(dat2_model)

##
total_time = 5
head(train, 10)
train %>% dim

Y = matrix(train$FreqIM, ncol=total_time, byrow = T) 
X1 = matrix(train$TypeCity, ncol=total_time, byrow = T)
X2 = matrix(train$TypeCounty, ncol=total_time, byrow = T)
X3 = matrix(train$TypeMisc, ncol=total_time, byrow = T)
X4 = matrix(train$TypeSchool, ncol=total_time, byrow = T)
X5 = matrix(train$TypeTown, ncol=total_time, byrow = T)
X6 = matrix(train$CoverageIM, ncol=total_time, byrow = T)
X7 = matrix(train$lnDeductIM, ncol=total_time, byrow = T)
X8 = matrix(train$NoClaimCreditIM, ncol=total_time, byrow = T)

X_list = list(X1, X2, X3, X4, X5, X6, X7, X8)

temp = alpha_est[1]
for(i in 1:length(X_list)){
   temp = temp + alpha_est[i+1]*X_list[[i]]
}
lambda_it  <- exp(temp)

temp = eta_est[1]
for(i in 1:length(X_list)){
   temp = temp + eta_est[i+1]*X_list[[i]]
}
zero_prob_it = 1/(1+exp(-temp))

ys = Y


set.seed(1)
m = 250
m_samp = sample(dim(ys)[1], size = m)


### BBVI
g_0 = 3.8
eta = 0.0

etas = c()
n_ELBOs = c()
detas = c()

ns=2000
n_update = 100
system.time(
   for (it in 1:n_update) {
      
      result = parameter_update_universal(1/(1000+it), eta)
      
      eta = result[[1]]
      n_ELBO = result[[2]]
      deta = result[[3]]
      
      etas[it] = eta
      n_ELBOs[it] = n_ELBO
      detas[it] = deta
      if(it %% 50 == 0){
         print(it)
      }
   }
)

# negative ELBO plot
plot(1:length(n_ELBOs), n_ELBOs, type = 'l')

g_qs = log(1+exp(etas))
# g_qs

mini = which.min(n_ELBOs) 
opt_g = g_qs[mini]

n_ELBOs[mini]
opt_g

g_qs
detas


library(coda)
library(rjags)
library(runjags)

ptm.init <- proc.time()
# data list
dataList=list(N=Y      , X1=X1, X2=X2, X3=X3, X4=X4, 
              M=nrow(Y), X5=X5, X6=X6, X7=X7, X8=X8,  
              alpha =  alpha_est,  eta= eta_est,
              c0=.001, d0=0.001)

# model
modelString="model {

## simulate R, N ############################# 
for (i in 1:M){
    theta[i] ~ dgamma(gam1, gam2)
    for(t in 1:5){
        R[i,t] ~ dbern( 1/(1+exp(-
                        eta[1]         - eta[2]*X1[i,t] -eta[3]*X2[i,t] -
                        eta[4]*X3[i,t] - eta[5]*X4[i,t] -eta[6]*X5[i,t] -
                        eta[7]*X6[i,t] - eta[2]*X7[i,t] -eta[9]*X8[i,t])) )
                        
        mu_N[i,t] = exp( alpha[1]        + alpha[2]*X1[i,t]+ alpha[3]*X2[i,t]+
                         alpha[4]*X3[i,t]+ alpha[5]*X4[i,t]+ alpha[6]*X5[i,t]+
                         alpha[7]*X6[i,t]+ alpha[8]*X7[i,t]+ alpha[9]*X8[i,t] )*theta[i]
        N[i,t] ~ dpois( mu_N[i,t]*(1-R[i,t]) + 0.00001 )
    }
}

## prior #####################################
invsigsq ~ dgamma(c0,d0)
gam1 = invsigsq; gam2 = invsigsq
}
"

nChains=3; nAdapt=5000; nUpdate=30000; nSamples=30000; nthin=5
ptm.init <- proc.time()
runJagsOut = run.jags(method="parallel", model=modelString, 
                      monitor=c('theta'),
                      data=dataList,
                      n.chains=nChains, adapt=nAdapt, burnin=nUpdate,
                      sample=ceiling(nSamples/nChains), thin=nthin,
                      summarise=FALSE, plots=FALSE)

codaSamples = as.mcmc.list(runJagsOut)

mcmcSamples=mcmc(codaSamples[[1]])
if( nChains > 1) for(ich in 2:nChains) mcmcSamples=rbind(mcmcSamples,mcmc(codaSamples[[ich]]))
para.samples=as.matrix(mcmcSamples)
time1=proc.time() - ptm.init

par1 = data.frame(para.samples, row.names=NULL) # 30000 * 500



Ytest <- test$FreqIM

temp = alpha_estt[1]
for(i in 1:length(X_list)){
   temp = temp + alpha_estt[i+1]*X_list[[i]]
}
lambd2_it  <- exp(temp)

cred_weight <- (rowSums(Y)+opt_g) / (rowSums((1-zero_prob_it)*lambda_it)+opt_g)
cre2_weight <- (rowSums(Y)+opt_g) / (rowSums(lambd2_it)+opt_g)
Bayes_weight <- colMeans(para.samples)

dzdz <- cbind(unique(train$PolicyNum), cred_weight, cre2_weight, Bayes_weight)
colnames(dzdz)[1] <- "PolicyNum"
test <- merge(test, dzdz, by="PolicyNum", all.x=TRUE)
test$cred_weight[is.na(test$cred_weight)] <- 1
test$cre2_weight[is.na(test$cre2_weight)] <- 1
test$Bayes_weight[is.na(test$Bayes_weight)] <- 1

naive_prm  <- predict(data_model, test)
cred_prm   <- naive_prm * test$cred_weight
Bayes_prm  <- naive_prm * test$Bayes_weight
naiv2_prm  <- predict(dat2_model, test, type="response")
cre2_prm  <- naiv2_prm * test$cre2_weight

naive_diff <- naive_prm - Ytest
cred_diff <-  cred_prm - Ytest
naiv2_diff <- naiv2_prm - Ytest
cre2_diff <-  cre2_prm - Ytest
Bayes_diff <- Bayes_prm - Ytest

naive_RMSE <- mean(naive_diff^2)
cred_RMSE <- mean( cred_diff^2)
naiv2_RMSE <- mean(naiv2_diff^2)
cre2_RMSE <- mean( cre2_diff^2)
Bayes_RMSE <- mean(Bayes_diff^2)

naive_MAE <- mean(abs(naive_diff))
cred_MAE <- mean(abs( cred_diff))
naiv2_MAE <- mean(abs(naiv2_diff))
cre2_MAE <- mean(abs( cre2_diff))
Bayes_MAE <- mean(abs(Bayes_diff))

round(c(naiv2_RMSE, cre2_RMSE, naive_RMSE, cred_RMSE, Bayes_RMSE),4)
round(c(naiv2_MAE , cre2_MAE , naive_MAE , cred_MAE , Bayes_MAE ),4)

library(knitr)
library(kableExtra)

estable <- cbind(summary(dat2_model)$coefficients[,c(1,4)],
  summary(data_model)$coefficients$zero[,c(1,4)],
  summary(data_model)$coefficients$count[,c(1,4)])
options(knitr.table.format = "latex")


kable(estable,digits=2,booktabs = T,
      linesep = c("", "", "","",  "","","", "","", "", "","\\hline"),
      bottomrule="\\hhline{=======}",escape = FALSE) %>% kable_styling(latex_options = c("hold_position", "scale_down")) %>%
   add_header_above(c(" "=1, "alpha" = 2, "eta" = 2, "alpha" = 2)) %>%
   add_header_above(c(" "=1, "No ZI" = 2,"With ZI" = 4))


