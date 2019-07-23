# Finite mixture with 2 components and Normal priors
#install.packages("rstanarm",dependencies = TRUE)
library(rstan)
library(shinystan)
library(rstanarm)
options(mc.cores = parallel::detectCores())
### The STAN model ###
rstan_options(auto_write = TRUE)

set.seed(689934)
df = read.csv("~/data.csv", header = TRUE, sep = "\t")
summary(df)

lmod = lm(df$Y ~ ., data = df)
summary(lmod)

lmod2 = lm(df$Y ~ df$var1 + df$var3, data = df)
summary(lmod2)

N <- length(df$Y)
y <- df$Y
x1 <- df$var1
x2 <- df$var2
x3 <- df$var3
K <- 2


stan_data <- list(y = y, N = N, K = K)

stan_code <- '
data {
  int<lower = 1> N;
  int<lower = 1> K;
  vector[N] y;
}

parameters {
  ordered[K] mu;
  vector<lower=0>[K] sigma;
  simplex[K] theta;
}

model {
  vector[K] log_theta = log(theta);
  sigma ~ cauchy(1,0.1);
  mu[1] ~ normal(0,0.5);
  mu[2] ~ normal(5,0.5);
  for (n in 1:N) {
    vector[K] lps = log_theta;
    for (k in 1:K)
      lps[k] +=normal_lpdf(y[n] | mu[k], sigma[k]);
    target += log_sum_exp(lps);
  }
}
'

degenerate_fit <- stan(model_code = stan_code, data=stan_data, chains=4, seed=483892929, refresh=2000)

print(degenerate_fit)
summary(degenerate_fit)
plot(degenerate_fit, show_density = TRUE, ci_level = 0.90, fill_color = "purple")

get_posterior_mean(degenerate_fit)

posterior <- extract(degenerate_fit, include = T)
yrep <- posterior_predict(posterior)
mean(apply(posterior$y_rep, 2, median) == y_rep)

library(shinystan)
launch_shinystan(degenerate_fit)

c_light_trans <- c("#DCBCBCBF")
c_light_highlight_trans <- c("#C79999BF")
c_mid_trans <- c("#B97C7CBF")
c_mid_highlight_trans <- c("#A25050BF")
c_dark_trans <- c("#8F2727BF")
c_dark_highlight_trans <- c("#7C0000BF")

params1 <- as.data.frame(extract(degenerate_fit, permuted=FALSE)[,1,])
params2 <- as.data.frame(extract(degenerate_fit, permuted=FALSE)[,2,])
params3 <- as.data.frame(extract(degenerate_fit, permuted=FALSE)[,3,])
params4 <- as.data.frame(extract(degenerate_fit, permuted=FALSE)[,4,])

par(mar = c(4, 4, 0.5, 0.5))
plot(params1$"mu[1]", params1$"mu[2]", col=c_dark_highlight_trans, pch=16, cex=0.8,
     xlab="mu1", xlim=c(-3, 10), ylab="mu2", ylim=c(-3, 10))
points(params2$"mu[1]", params2$"mu[2]", col=c_dark_trans, pch=16, cex=0.8)
points(params3$"mu[1]", params3$"mu[2]", col=c_mid_highlight_trans, pch=16, cex=0.8)
points(params4$"mu[1]", params4$"mu[2]", col=c_mid_trans, pch=16, cex=0.8)
lines(0.08*(1:100) - 1, 0.08*(1:100) - 1, col="grey", lw=2)
legend("topright", c("Chain 1", "Chain 2", "Chain 3", "Chain 4"),
       fill=c(c_dark_highlight_trans, c_dark_trans,
              c_mid_highlight_trans, c_mid_trans), box.lty=0, inset=0.0005)
