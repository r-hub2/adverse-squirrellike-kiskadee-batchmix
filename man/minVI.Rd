% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/minVI.R
\name{minVI}
\alias{minVI}
\title{Minimium VI}
\usage{
minVI(psm, cls.draw = NULL, method = "avg", max.k = NULL)
}
\arguments{
\item{psm}{The posterior similarity matrix for a set of clustering MCMC
samples such as is returned by the `createSimilarityMat` function.}

\item{cls.draw}{The set of clustering MCMC samples used to generate `psm`.
Only required if `method` is one of `'draws'` or `'all'`.}

\item{method}{String indicating which method is used to find the point
estimate clustering. Must be one of `'avg'`, `'comp'`, `'draws'` or `'all'`.
Defaults to `'avg'`. If `'all'` is passed the three methods are all applied
to return different choices of point clustering.}

\item{max.k}{The maximum number of clusters to consider. Only used by the
`'comp'` and `'avg'` methods. Defaults to one-quarter the number of data
points rounded up.}
}
\value{
If `method` is `'all'` returns a matrix of four clusterings, one for
each method and a repeat of that which performs best based on minimising the
Variation of Information between the clustering and the PSM. Otherwise
returns a vector. This is annotated with the attribute `"info"`, a named list
describing:

* `.$loss`: the loss score used (Variation of Information)

* `.$maxNClusters`: the `max.k` value used by the `'comp'` and `'avg'` methods

* `.$expectedLoss`: the estimated minimum Variation of Information for the point
clustering(s)

* `.$method`: the point method used to infer the clustering(s)

Names are due to legacy reasons - this function is replacing the
`salso::salso` function and name choices are to minimise workflow damage.
}
\description{
Local implementation of S. Wade's `minVI` function from their
`mcclust.ext` package (available from github).
Reimplemented here to avoid dependency on a non-CRAN package and we have
dropped the `greedy` method. Finds the optimal partition by minimising the
lower bound to the Variation of Information obtained from Jensen's inequality
where the expectation and log are reversed. For full details please see the
aforementioned package and Wade and Ghahramani, 2018, 'Bayesian Cluster
Analysis: Point Estimation and Credible Balls (with Discussion)'.
}
\examples{
\dontrun{
# MCMC samples and BIC vector
mcmc_outputs <- runMCMCChains(
  X,
  n_chains,
  R,
  thin,
  batch_vec,
  type
)

# Note that in this toy example we have not applied a burn in
psm <- createSimilarityMat(mcmc_outputs[[1]]$samples)
cl_est <- minVI(psm, mcmc_outputs[[1]]$samples)
}
}
