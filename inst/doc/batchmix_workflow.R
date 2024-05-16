## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)

set.seed(1)

## ----dataGen------------------------------------------------------------------
library(ggplot2)
library(batchmix)

# Data dimensions
N <- 600
P <- 4
K <- 5
B <- 7

# Generating model parameters
mean_dist <- 2.25
batch_dist <- 0.3
group_means <- seq(1, K) * mean_dist
batch_shift <- rnorm(B, mean = batch_dist, sd = batch_dist)
std_dev <- rep(2, K)
batch_var <- rep(1.2, B)
group_weights <- rep(1 / K, K)
batch_weights <- rep(1 / B, B)
dfs <- c(4, 7, 15, 60, 120)

my_data <- generateBatchData(
  N,
  P,
  group_means,
  std_dev,
  batch_shift,
  batch_var,
  group_weights,
  batch_weights,
  type = "MVT",
  group_dfs = dfs
)

## ----dataClean----------------------------------------------------------------
X <- my_data$observed_data

true_labels <- my_data$group_IDs
fixed <- my_data$fixed
batch_vec <- my_data$batch_IDs

alpha <- 1
initial_labels <- generateInitialLabels(alpha, K, fixed, true_labels)

## ----runMCMCChains------------------------------------------------------------
# Sampling parameters
R <- 1000
thin <- 50
n_chains <- 4

# Density choice
type <- "MVT"

# MCMC samples and BIC vector
mcmc_output <- runMCMCChains(
  X,
  n_chains,
  R,
  thin,
  batch_vec,
  type,
  initial_labels = initial_labels,
  fixed = fixed
)

## ----plotAcceptanceRatesEarly-------------------------------------------------
plotAcceptanceRates(mcmc_output)

## ----likelihood---------------------------------------------------------------
plotLikelihoods(mcmc_output)

## ----continueChains-----------------------------------------------------------
R_new <- 9000

# Given an initial value for the parameters
new_output <- continueChains(
  mcmc_output,
  X,
  fixed,
  batch_vec,
  R_new,
  keep_old_samples = TRUE
)

## ----continuedLikelihood------------------------------------------------------
plotLikelihoods(new_output)

## ----plotAcceptanceRates------------------------------------------------------
plotAcceptanceRates(new_output)

## ----processChains------------------------------------------------------------
# Burn in
burn <- 5000

# Process the MCMC samples
processed_samples <- processMCMCChains(new_output, burn)

## ----pca----------------------------------------------------------------------
chain_used <- processed_samples[[1]]

pc <- prcomp(X, scale = T)
pc_batch_corrected <- prcomp(chain_used$inferred_dataset)

plot_df <- data.frame(
  PC1 = pc$x[, 1],
  PC2 = pc$x[, 2],
  PC1_bf = pc_batch_corrected$x[, 1],
  PC2_bf = pc_batch_corrected$x[, 2],
  pred_labels = factor(chain_used$pred),
  true_labels = factor(true_labels),
  prob = chain_used$prob,
  batch = factor(batch_vec)
)

plot_df |>
  ggplot(aes(
    x = PC1,
    y = PC2,
    colour = true_labels,
    alpha = prob
  )) +
  geom_point()

plot_df |>
  ggplot(aes(
    x = PC1_bf,
    y = PC2_bf,
    colour = pred_labels,
    alpha = prob
  )) +
  geom_point()

test_inds <- which(fixed == 0)

sum(true_labels[test_inds] == chain_used$pred[test_inds]) / length(test_inds)

