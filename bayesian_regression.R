# Bayesian Regression with full model
library(rstan)
library(shinystan)
options(mc.cores = parallel::detectCores())
### The STAN model ###
rstan_options(auto_write = TRUE)
#install.packages("mvtnorm",dependencies = TRUE)
library(mvtnorm)

set.seed(689934)
df = read.csv("~/data.csv", header = TRUE, sep = "\t")
summary(df)

lmod = lm(df$Y ~ ., data = df)
summary(lmod)
#plot(lmod)

r = residuals(lmod)

#plot(density((r-mean(r))/sqrt(var(r))), xlim=range(c(-3,3)), ylim=range(c(0,0.4)))
#plot(density(r))

N <- length(df$Y)
y <- df$Y
x1 <- df$var1
x2 <- df$var2
x3 <- df$var3

# Model matrix (with column of 1s for intercept and three covariates)
X <- cbind(Const = 1, X1 = x1, X2 = x2, X3 = x3)
K <- ncol(X)

stan_code <- '
data {
  int           N ; // integer, number of observations
  int           K ; // integer, number of columns in model matrix
  matrix[N,K]   X ; // N by K model matrix
  vector[N]     y ; // vector of N observations
}

parameters {
  real<lower=0> sigma ; // real number > 0, standard deviation
  vector[K]     beta ;  // K-vector of regression coefficients
}

model {
  beta ~ normal(0, 5) ;       // prior for betas
  sigma ~ cauchy(1, 0.1) ;    // prior for sigma
  y ~ normal(X*beta, sigma) ; // vectorized likelihood
}

generated quantities {
// Here we do the simulations from the posterior predictive distribution

  vector[N] y_rep_reg ; // vector of same length as the data y
  vector[N] y_rep_normal;
  vector[N] y_rep_err;
  vector[N] y_poi;
  for (n in 1:N) {
    y_rep_normal[n] = normal_rng(X[n]*beta, sigma);
    y_rep_reg[n] = X[n]*beta;
    y_rep_err[n] = X[n]*beta + normal_rng(0, sigma);
    y_poi[n] = gamma_rng(20,1);
  }

}
'

# Prepare the data we'll need as a list
stan_data <- list(y = y, X = X, N = N, K = K)

# Fit the model
stanfit <- stan(model_code = stan_code, data = stan_data, iter=10000, chains=4, seed=483892929, refresh=2000)

summary(stanfit)
get_posterior_mean(stanfit)

posterior <- extract(stanfit, include = T)
#yrep <- posterior_predict(posterior)
mean(apply(posterior$y_rep_normal, 2, median) == y)
mean(apply(posterior$y_rep_reg, 2, median) == y)
mean(apply(posterior$y_rep_err, 2, median) == y)

truehist(y, col="#B2001D")
lines(density(y), col="skyblue", lwd=2)
summary(y)

truehist(posterior$y_rep_normal, 50, col="#B2001D")
lines(density(posterior$y_rep_normal), col="skyblue", lwd=2)
lines(density(y), col="green", lwd=2)
summary(posterior$y_rep_normal[,1])

truehist(posterior$y_rep_reg, 50, col="#B2001D")
lines(density(posterior$y_rep_reg), col="skyblue", lwd=2)
lines(density(y), col="green", lwd=2)
summary(posterior$y_rep_reg[,1])

truehist(posterior$y_rep_err, 50, col="#B2001D")
lines(density(posterior$y_rep_err), col="skyblue", lwd=2)
lines(density(y), col="green", lwd=2)
summary(posterior$y_rep_err[,1])

truehist(posterior$y_poi, 50, col="#B2001D")
lines(density(posterior$y_poi), col="skyblue", lwd=2)
lines(density(y), col="green", lwd=2)
summary(posterior$y_poi[,1])

#posterior$y_rep

# Launch ShinyStan
launch_shinystan(stanfit)
