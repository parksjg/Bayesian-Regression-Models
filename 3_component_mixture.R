# Finite mixture with 3 components and Normal priors

library(rstan)
library(shinystan)
library(MASS)
options(mc.cores = parallel::detectCores())
### The STAN model ###
rstan_options(auto_write = TRUE)

set.seed(689934)
df = read.csv("~/data.csv", header = TRUE, sep = "\t")
summary(df)

lmod = lm(df$Y ~ ., data = df)
summary(lmod)
par(mfrow = c(2,2))
plot(lmod)

par(mfrow = c(1,1))
r = residuals(lmod)

#plot(density((r-mean(r))/sqrt(var(r))), xlim=range(c(-3,3)), ylim=range(c(0,0.4)))
plot(density(r))

N <- length(df$Y)
y <- df$Y
x1 <- df$var1
x2 <- df$var2
x3 <- df$var3
K <- 3
#theta <- c(1/3, 1/3, 1/3)
#sigma <- c(1.433017, 1.013836, 0.9755867)
#sigma <- c(1.4, 1, .98)
#sigma <- 0.1

X <- cbind(Const = 1, X1 = x1, X2 = x2, X3 = x3)
J <- ncol(X)

# Prepare the data we'll need as a list
stan_data <- list(y = y, N = N, J = J, K = K, X = X)

stan_code <- '
data {
  int<lower = 1> N;  // integer, number of observations
  int<lower = 1> J;  // integer, number of columns in model matrix
  int<lower = 1> K;
  matrix[N,J]   X ; // N by J model matrix
  vector[N] y;
}

parameters {
  vector<lower=0>[K] sigma;
  ordered[K] mu;
  real<lower=0> tao ; // real number > 0, standard deviation
  vector[J]     beta ;  // J-vector of regression coefficients
  simplex[K] theta;
}

model {
  vector[K] log_theta = log(theta);
  sigma[1] ~ cauchy(1,.1);
  sigma[2] ~ cauchy(1,.1);
  sigma[3] ~ cauchy(1,.1);
  mu[1] ~ normal(-0.06782,.1);
  mu[2] ~ normal(0.01622,.1);
  mu[3] ~ normal(4.895,.1);
  beta ~ normal(0, 5) ;       // prior for betas
  tao ~ cauchy(1, 0.1) ;    // prior for y sigma
  y ~ normal(X*beta, tao) ; // vectorized likelihood
  for (n in 1:N) {
    vector[K] lps = log_theta;
    for (k in 1:K)
      lps[k] +=normal_lpdf(y[n] | mu[k], sigma[k]);
      //lps[k] +=lognormal_lpdf(y[n] | mu[k], sigma[k]);
      //lps[k] +=student_t_lpdf(y[n] | 117, mu[k], sigma[k]);
    target += log_sum_exp(lps);
  }
}

generated quantities{
  vector[N] y_rep ; // vector of same length as the data y
  vector[N] y_rep_reg ; // vector of same length as the data y
  vector[N] y_rep_normal;
  vector[N] y_rep_err;
  vector[N] y_gam;
  vector[K] log_theta = log(theta);
 {
  for (n in 1:N) {
    vector[K] lps = log_theta;
    for (k in 1:K)
      lps[k] =normal_lpdf(y[n] | X[n]*beta, tao);
    y_rep[n] = log_sum_exp(lps);
    
  }
  
  for (n in 1:N) {
    y_rep_normal[n] = normal_rng(X[n]*beta, tao);
    y_rep_reg[n] = X[n]*beta;
    y_rep_err[n] = X[n]*beta + normal_rng(0, tao);
    y_gam[n] = gamma_rng(20,1);
  }
 }
}
'

degenerate_fit <- stan(model_code = stan_code, data=stan_data, iter=10000, chains=4, seed=483892929, refresh=2000)

plot(degenerate_fit, show_density = TRUE, ci_level = 0.5, fill_color = "purple")

#pairs(degenerate_fit)
pairs(df)

print(degenerate_fit)
summary(degenerate_fit)

get_posterior_mean(degenerate_fit)
#get_logposterior(degenerate_fit)

posterior <- extract(degenerate_fit, include = T)
mean(apply(posterior$y_rep_reg, 2, median) == y)
mean(posterior$y_rep_reg[,1] == y)
truehist(posterior$y_rep_reg[2,])

truehist(y, col="#B2001D")
lines(density(y), col="skyblue", lwd=2)
summary(y)


truehist(posterior$y_rep_reg, 50, col="#B2001D")
lines(density(posterior$y_rep_reg), col="skyblue", lwd=2)
lines(density(y), col="green", lwd=2)
summary(posterior$y_rep_reg[,1])
mean(posterior$y_rep_reg)
mean(df$Y)

truehist(posterior$y_rep_normal[,3],col="#B2001D", main="Posterior Predictive", xlab="y_rep_normal ~ N(X*beta, tao)")
lines(density(posterior$y_rep_normal[13873,]), col="skyblue", lwd=2)
lines(density(posterior$y_rep_normal), col="skyblue", lwd=2)
lines(density(y), col="green", lwd=2)
summary(posterior$y_rep_normal[,1])
#plot(posterior$y_rep_normal[3,])

truehist(posterior$y_rep[,2], col="#B2001D")
lines(density(posterior$y_rep), col="skyblue", lwd=2)
lines(density(y), col="green", lwd=2)
summary(posterior$y_rep[1,])

truehist(posterior$y_rep_err[,3], col="#B2001D", main="Posterior Predictive 2", xlab="y_rep_err ~ X*beta + N(0, tao)")
lines(density(posterior$y_rep_err), col="blue", lwd=2)
lines(density(posterior$y_rep_err[7873,]), col="skyblue", lwd=2)
lines(density(posterior$y_rep_err[19326,]), col="skyblue", lwd=2)
lines(density(y), col="green", lwd=2)
summary(posterior$y_rep_err[,8])

mseNomral = mean((df$Y - posterior$y_rep_normal)^2)
mseNormal

mseErr = mean((df$Y - posterior$y_rep_err)^2)
mseErr


truehist(y, col="#B2001D")
lines(density(y), col="skyblue", lwd=2)
summary(y)

truehist(x1, col="#C2001D")
lines(density(x1), col="skyblue", lwd=2)
summary(x1)

truehist(x2, col="#B97C7CBF")
lines(density(x2), col="skyblue", lwd=2)
summary(x2)

truehist(x3, col="#7C0000BF")
lines(density(x3), col="skyblue", lwd=2)
summary(x3)

launch_shinystan(degenerate_fit)

# New sampling and fitting
s_model = stan_model(model_code = stan_code)
fit <- sampling(s_model, data = stan_data)
summary(fit)
get_posterior_mean(fit)
vb_fit = vb(s_model, data=stan_data, iter=10000)

m <- stan_model(model_code = stan_code)
f <- optimizing(m, hessian = TRUE)
f
