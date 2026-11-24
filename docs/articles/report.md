# 1 Introduction

# 1 Introduction

Counts such as the number of doctor visits are often **overdispersed**:
their variance is larger than their mean, which the Poisson model does
not allow. They often also contain **more zeros** than a Poisson (or
even a negative binomial) distribution predicts. The negative binomial
(NB2) model handles overdispersion through a dispersion parameter
$`\theta`$. The zero-inflated negative binomial (ZINB) model also allows
a separate process that produces “structural” zeros.

The **nbzinbKoo** package fits both models with algorithms written from
scratch: iteratively reweighted least squares (IRLS) alternated with
Newton–Raphson for NB regression, and an EM algorithm whose M-steps are
IRLS fits for ZINB regression. It does not call `optim()` or any other
general-purpose optimizer. This report describes the methods (Section 2)
and the package (Section 3), checks the results against `MASS::glm.nb()`
and `pscl::zeroinfl()` (Section 4), studies the ZINB estimator by
simulation (Section 5), and analyzes physician office visits in the
NMES1988 data (Section 6).

# 2 Methods

## 2.1 Negative binomial regression

For counts $`y_i`$ with covariates $`x_i`$ and offset $`o_i`$, the NB2
model has $`\log\mu_i = \eta_i = x_i^\top\beta + o_i`$ and
$`\operatorname{Var}(Y_i) = \mu_i + \mu_i^2/\theta`$. With prior weights
$`a_i`$ (all 1 in an ordinary fit) the log-likelihood is
``` math
\ell(\beta,\theta) = \sum_i a_i\Big[\log\Gamma(y_i+\theta) - \log\Gamma(\theta) - \log y_i!
  + \theta\log\theta + y_i\log\mu_i - (\theta+y_i)\log(\theta+\mu_i)\Big].
```

**IRLS for $`\beta`$ ($`\theta`$ fixed).** Fisher scoring for a log-link
GLM with variance $`V(\mu) = \mu + \mu^2/\theta`$ is weighted least
squares with
``` math
w_i = a_i\frac{(\partial\mu_i/\partial\eta_i)^2}{V(\mu_i)} = a_i\frac{\theta\mu_i}{\theta+\mu_i}, \qquad
z_i = (\eta_i - o_i) + \frac{y_i-\mu_i}{\mu_i}, \qquad
\beta^{(t+1)} = (X^\top WX)^{-1}X^\top Wz .
```
The least-squares step is solved by a QR decomposition of $`W^{1/2}X`$,
and a step that lowers $`\ell`$ is halved.

**Newton–Raphson for $`\theta`$ ($`\beta`$ fixed).** With
$`U(\theta) = \sum_i a_i[\psi(y_i+\theta) - \psi(\theta) + \log\theta + 1 - \log(\theta+\mu_i) - (y_i+\theta)/(\theta+\mu_i)]`$
and
$`H(\theta) = \sum_i a_i[\psi'(y_i+\theta) - \psi'(\theta) + 1/\theta - 2/(\theta+\mu_i) + (y_i+\theta)/(\theta+\mu_i)^2]`$,
we update $`\phi = \log\theta`$, which keeps $`\theta>0`$:
$`\phi \leftarrow \phi - U_\phi/H_\phi`$ with $`U_\phi = \theta U`$ and
$`H_\phi = \theta^2H + \theta U`$.

**Outer loop and standard errors.** We start from a Poisson IRLS fit and
the moment estimate
$`\theta^{(0)} = \sum_i a_i / \sum_i a_i(y_i/\mu_i - 1)^2`$, then
alternate the two steps until $`\ell`$ changes by less than $`10^{-8}`$.
At convergence
$`\widehat{\operatorname{Var}}(\hat\beta) = (X^\top\hat WX)^{-1}`$
(treating $`\theta`$ as known, as `glm.nb()` does) and
$`\operatorname{SE}(\hat\theta) = \{-H(\hat\theta)\}^{-1/2}`$.

## 2.2 Zero-inflated negative binomial regression

The ZINB model mixes a point mass at zero (probability $`\pi_i`$,
$`\operatorname{logit}\pi_i = z_i^\top\gamma`$) with an
NB$`(\mu_i,\theta)`$ count distribution. Its observed-data
log-likelihood is
``` math
\ell(\beta,\gamma,\theta) = \sum_{y_i=0}\log\big[\pi_i + (1-\pi_i)f_{NB}(0;\mu_i,\theta)\big]
 + \sum_{y_i>0}\big[\log(1-\pi_i) + \log f_{NB}(y_i;\mu_i,\theta)\big],
```
with $`f_{NB}(0;\mu,\theta) = \{\theta/(\theta+\mu)\}^\theta`$. Because
of the log of a sum, this is not an exponential-family likelihood and
IRLS cannot be applied to it directly. We maximize it by EM, treating
the indicator $`D_i`$ of a structural zero as missing data.

- **E-step.**
  $`\tau_i = P(D_i = 1\mid y_i) = \pi_i/\{\pi_i + (1-\pi_i)f_{NB}(0;\mu_i,\theta)\}`$
  if $`y_i = 0`$ and $`\tau_i = 0`$ otherwise.
- **M-step, zero part.** Maximize
  $`\sum_i[\tau_i\log\pi_i + (1-\tau_i)\log(1-\pi_i)]`$: a logistic
  regression with fractional response $`\tau_i`$, fitted by IRLS with
  $`w_i = \pi_i(1-\pi_i)`$ and
  $`z_i = z_i^\top\gamma + (\tau_i - \pi_i)/\{\pi_i(1-\pi_i)\}`$.
- **M-step, count part.** Maximize
  $`\sum_i(1-\tau_i)\log f_{NB}(y_i;\mu_i,\theta)`$: exactly the NB fit
  above with prior weights $`a_i = 1-\tau_i`$ (the package reuses the
  same code).

EM starts from the NB fit for $`(\beta,\theta)`$ and a logistic IRLS fit
of $`I(y_i=0)`$ on $`z_i`$ for $`\gamma`$, and stops when $`\ell`$
changes by less than $`10^{-8}`$. The package records $`\ell`$ at every
iteration and warns if it ever decreases.

**Standard errors.** We invert the observed information of $`\ell`$ in
$`(\beta,\gamma,\phi=\log\theta)`$, computed **analytically**. For
observation $`i`$ let $`h_i = \log f_{NB}(y_i;\mu_i,\theta)`$ and
$`\rho_i = 1-\tau_i`$. Then
``` math
\frac{\partial\ell_i}{\partial\eta_{z}} = \tau_i - \pi_i,\quad
\frac{\partial\ell_i}{\partial a} = \rho_i h_a,\quad
\frac{\partial^2\ell_i}{\partial\eta_z^2} = \tau_i(1-\tau_i) - \pi_i(1-\pi_i),\quad
\frac{\partial^2\ell_i}{\partial\eta_z\,\partial a} = -\tau_i\rho_i h_a,\quad
\frac{\partial^2\ell_i}{\partial a\,\partial b} = \rho_i h_{ab} + \tau_i\rho_i h_a h_b,
```
for $`a,b\in\{\eta_c,\phi\}`$, where
$`h_{\eta} = \theta(y-\mu)/(\theta+\mu)`$,
$`h_{\eta\eta} = -\theta\mu(\theta+y)/(\theta+\mu)^2`$,
$`h_\phi = \theta U_i`$, $`h_{\phi\phi} = \theta^2H_i + \theta U_i`$ and
$`h_{\eta\phi} = \theta\mu(y-\mu)/(\theta+\mu)^2`$. The chain rule
($`\partial\eta_c/\partial\beta = x_i`$,
$`\partial\eta_z/\partial\gamma = z_i`$) gives the blocks of the
Hessian. The test suite checks this Hessian against
`numDeriv::hessian()` of the package’s own log-likelihood.

# 3 The R package

The package is installed with
`devtools::install_github("gracko95/nbzinbKoo")`. Its main functions
are:

| Function | Purpose |
|----|----|
| `nbreg(formula, data, offset, control)` | NB2 regression (IRLS + Newton–Raphson) |
| `zinbreg(formula, data, offset, control)` | ZINB regression (EM with IRLS M-steps); formula `y ~ x | z` |
| `nbreg_control()`, `zinbreg_control()` | tolerances, maximum iterations, `trace`, Hessian type |

Both functions return S3 objects (classes `"nbreg"` and `"zinbreg"`)
with methods `print`, `summary`, `coef`, `vcov`, `logLik` (so `AIC()`
and `BIC()` work), `fitted`, `residuals` (response and Pearson), `nobs`
and `predict` (with `newdata`; for `zinbreg`, `type = "response"`,
`"count"`, `"zero"` or `"prob0"`). The code is organized as `R/nbreg.R`
(NB engine), `R/zinbreg.R` (EM engine and analytic Hessian),
`R/methods.R` and `R/utils.R` (log-likelihood, $`\theta`$ derivatives,
weighted least squares).

``` r
fit <- zinbreg(art ~ fem + kid5 + ment | ment, data = bioChemists)
round(coef(fit), 3)
```

    count_(Intercept)    count_femWomen        count_kid5        count_ment 
                0.494            -0.228            -0.131             0.024 
     zero_(Intercept)         zero_ment 
               -0.795            -0.619 

``` r
c(theta = fit$theta, logLik = fit$loglik, EM_iterations = fit$iter)
```

            theta        logLik EM_iterations 
         2.695777  -1554.752519    113.000000 

# 4 Benchmarking

We compare `nbreg()` with `MASS::glm.nb()` and `zinbreg()` with
`pscl::zeroinfl(dist = "negbin")` on `bioChemists` (Long, 1997) and
`NMES1988` (Deb and Trivedi, 1997). The reference functions were run
with tight tolerances (`epsilon = 1e-10` and `reltol = 1e-12`).
[Table 1](#tbl-agree) gives the maximum absolute differences, which are
all far below the required $`10^{-5}`$ (NB coefficients and SEs),
$`10^{-4}`$ (NB $`\theta`$ and log-likelihood) and $`10^{-3}`$ (ZINB). A
positive log-likelihood difference means that our fit attains a slightly
higher maximum.

| Fit               |    Coef |      SE |   theta |   logLik | Iter | Time (ours) | Time (ref.) |
|:------------------|--------:|--------:|--------:|---------:|-----:|------------:|------------:|
| NB, bioChemists   | 2.1e-08 | 9.8e-10 | 6.7e-08 |  4.5e-13 |    4 |       0.010 |       0.011 |
| NB, NMES1988      | 4.7e-08 | 9.2e-10 | 4.3e-08 |  7.1e-11 |    3 |       0.049 |       0.042 |
| ZINB, bioChemists | 1.5e-04 | 9.4e-05 | 1.3e-05 | -4.2e-08 |  111 |       0.708 |       0.028 |
| ZINB, NMES1988    | 2.6e-05 | 1.7e-05 | 4.4e-06 | -1.6e-08 |   53 |       1.989 |       0.127 |

Table 1: Maximum absolute differences from the reference
implementations, iterations of our algorithm (outer NB iterations or EM
iterations), and median elapsed time in seconds. For ZINB the theta row
is log(theta).

| Term                  |  zinbreg | zeroinfl | SE zinbreg | SE zeroinfl |
|:----------------------|---------:|---------:|-----------:|------------:|
| count\_(Intercept)    |  1.19372 |  1.19371 |    0.05666 |     0.05666 |
| count_hospital        |  0.20148 |  0.20148 |    0.02036 |     0.02036 |
| count_healthpoor      |  0.28513 |  0.28513 |    0.04509 |     0.04509 |
| count_healthexcellent | -0.31934 | -0.31934 |    0.06040 |     0.06040 |
| count_chronic         |  0.12900 |  0.12900 |    0.01193 |     0.01193 |
| count_gendermale      | -0.08028 | -0.08028 |    0.03102 |     0.03102 |
| count_school          |  0.02142 |  0.02142 |    0.00436 |     0.00436 |
| count_insuranceyes    |  0.12586 |  0.12586 |    0.04159 |     0.04159 |
| zero\_(Intercept)     | -0.04689 | -0.04688 |    0.26856 |     0.26856 |
| zero_hospital         | -0.80046 | -0.80048 |    0.42080 |     0.42082 |
| zero_chronic          | -1.24791 | -1.24794 |    0.17831 |     0.17832 |
| zero_gendermale       |  0.64768 |  0.64768 |    0.20011 |     0.20011 |
| zero_school           | -0.08377 | -0.08377 |    0.02625 |     0.02625 |
| zero_insuranceyes     | -1.17561 | -1.17562 |    0.22012 |     0.22012 |
| log(theta)            |  0.39415 |  0.39414 |    0.03504 |     0.03504 |

Table 2: ZINB estimates and standard errors for NMES1988: zinbreg()
versus zeroinfl().

The side-by-side tables for the other three fits are in the Appendix
([Table 7](#tbl-app-nb-bio), [Table 8](#tbl-app-nb-nmes) and
[Table 9](#tbl-app-zinb-bio)). EM needs more iterations than BFGS and is
slower (each M-step is itself an iterative fit), but every run converged
and the log-likelihood never decreased.

# 5 Simulation study

We simulated $`R = 200`$ data sets of size $`n = 500`$ (seed 651) from a
ZINB model with $`\log\mu_i = 1 + 0.5x_{1i}  -0.7x_{2i}`$
($`x_1\sim N(0,1)`$, $`x_2\sim\text{Bernoulli}(0.5)`$),
$`\operatorname{logit}\pi_i = -1 + 0.8z_{1i}`$ ($`z_1\sim N(0,1)`$) and
$`\theta = 1.5`$, and fitted each with `zinbreg()` (script
`inst/benchmark/simulation.R`). 200 of 200 fits converged.

| Parameter          |   True | Mean est. |   Bias | Mean SE | MC SD | Coverage |
|:-------------------|-------:|----------:|-------:|--------:|------:|---------:|
| count\_(Intercept) |  1.000 |     0.997 | -0.003 |   0.099 | 0.101 |    0.930 |
| count_x1           |  0.500 |     0.502 |  0.002 |   0.064 | 0.064 |    0.960 |
| count_x2           | -0.700 |    -0.709 | -0.009 |   0.126 | 0.131 |    0.945 |
| zero\_(Intercept)  | -1.000 |    -1.053 | -0.053 |   0.329 | 0.335 |    0.960 |
| zero_z1            |  0.800 |     0.840 |  0.040 |   0.235 | 0.246 |    0.955 |
| log(theta)         |  0.405 |     0.444 |  0.039 |   0.252 | 0.248 |    0.965 |

Table 3: Simulation results: bias, mean estimated SE versus Monte Carlo
SD of the estimates, and coverage of 95% Wald intervals.

Based on the results of the simulation, we can see that the bias is much
smaller (about a less than a fifth) relative to the SE. The mean SEs are
very close to the Monte Carlo SDs. For the coverage rate, all six values
fall within 0.95 ± 2 × 0.015, which is roughly 0.92–0.98. So, the
deviations are what you’d expect from just 200 replications. Thus, the
simulation demonstrates that our function zinbreg() recovers the true
parameters pretty closely and the analytic standard errors can be
trusted.

# 6 Data analysis: physician office visits (NMES1988)

The NMES1988 data record the number of physician office visits
(`visits`) for 4406 people aged 66 and over. The mean is 5.77 but the
variance is 45.7, about 8 times larger, which is strong evidence of
overdispersion. 15.5% of the people made no visits, while a Poisson
distribution with the same mean predicts only 0.31%
([Figure 1](#fig-visits)).

![](report_files/figure-html/fig-visits-1.png)

Figure 1: Observed distribution of visits (bars, 30 = 30 or more) and
frequencies expected under a Poisson distribution with the same mean
(points).

``` r
f_count <- visits ~ hospital + health + chronic + gender + school + insurance
m_pois <- glm(f_count, family = poisson, data = NMES1988)      # reference only
m_nb   <- nbreg(f_count, data = NMES1988)
m_zinb <- zinbreg(visits ~ hospital + health + chronic + gender + school + insurance |
                    hospital + chronic + gender + school + insurance, data = NMES1988)
```

| Model   |  df |   logLik |     AIC |     BIC | Expected zeros |
|:--------|----:|---------:|--------:|--------:|---------------:|
| Poisson |   8 | -17971.6 | 35959.2 | 36010.4 |           46.7 |
| NB      |   9 | -12170.6 | 24359.1 | 24416.6 |          608.0 |
| ZINB    |  15 | -12090.7 | 24211.4 | 24307.3 |          708.9 |

Table 4: Model comparison for NMES1988, including the expected number of
zeros under each model.

The data contain 683 zeros.

| Covariate       |  NB IRR (95% CI)  | ZINB count IRR (95% CI) |
|:----------------|:-----------------:|:-----------------------:|
| hospital        | 1.24 (1.20, 1.29) |    1.22 (1.18, 1.27)    |
| healthpoor      | 1.36 (1.23, 1.49) |    1.33 (1.22, 1.45)    |
| healthexcellent | 0.71 (0.63, 0.80) |    0.73 (0.65, 0.82)    |
| chronic         | 1.19 (1.16, 1.22) |    1.14 (1.11, 1.16)    |
| gendermale      | 0.88 (0.83, 0.94) |    0.92 (0.87, 0.98)    |
| school          | 1.03 (1.02, 1.04) |    1.02 (1.01, 1.03)    |
| insuranceyes    | 1.25 (1.16, 1.35) |    1.13 (1.05, 1.23)    |

Table 5: Count part: incidence rate ratios (IRR) with 95% Wald
confidence intervals.

| Covariate    |    OR (95% CI)    | p-value |
|:-------------|:-----------------:|:-------:|
| hospital     | 0.45 (0.20, 1.02) | 0.0571  |
| chronic      | 0.29 (0.20, 0.41) | \<1e-04 |
| gendermale   | 1.91 (1.29, 2.83) | 0.0012  |
| school       | 0.92 (0.87, 0.97) | 0.0014  |
| insuranceyes | 0.31 (0.20, 0.48) | \<1e-04 |

Table 6: ZINB zero part: odds ratios (OR) of being a structural zero (a
person who would not visit a physician), with 95% Wald confidence
intervals.

The NB model estimates $`\hat\theta = 1.21`$ (SE 0.034), and the ZINB
model $`\hat\theta = 1.48`$.

The NMES1988 dataset has a variance of 45.7, which is about eight times
the mean, which is 5.77. Thus, we can notice that there is an
overdispersion problem. The ZINB and NB model actually takes into
account the overdispersion problem this dataset has, which is why both
of them perform much better compared to the Poisson model. We can
actually see that the Poisson model expects much less zeros, because it
assumes equal mean and variance. The ZINB model takes into account the
excess zero problem, which can be evidently seen in
[Figure 1](#fig-visits). Thus, the expected zeros for the ZINB model is
the closest to the number of zeros the data actually contains (708.9 vs
683) Based on the [Table 4](#tbl-compare), we can see that the ZINB
model has a lower AIC and BIC and favor the ZINB model over the NB one,
so they are in agreement.

Looking at [Table 5](#tbl-irr), we can see the incidence rate ratios
with 95% Wald confidence intervals. These results describe people who do
visit a physician, holding the other covariates fixed (count part). We
can see that the strongest effects for increased physicians visits are
hospital stays, health status, and chronic conditions. The weaker
effects are insurance status, gender, and school. For the NB model,
holding other variables fixed:

- hospital: each additional hospital stay multiplies the expected number
  of physician office visits by 1.24, or a 24% increase, with 95%
  confidence that the factor is between 1.20 and 1.29.

- healthpoor: individuals with poor health are expected to visit the
  physician office by a factor of 1.36, or 36% more than individuals
  with average health, with a 95% confidence that the factor is between
  1.23 and 1.49.

- healthexcellent: individuals with excellent health are expected to
  visit the physician office by a factor of 0.71, or 29% less than
  individuals with average health, with a 95% confidence that the factor
  is between 0.63 and 0.80.

- chronic: each additional chronic condition multiplies the expected
  number of physician office visits by 1.19, or a 19% increase, with a
  95% confidence that the factor is between 1.16 and 1.22.

- gendermale: men are expected to visit the physician office by a factor
  of 0.88, or 12% less compared to women, with a 95% confidence that the
  factor is between 0.83 and 0.94.

- school: each additional year of education multiplies the expected
  number of physician office visits by 1.03, or a 3% increase, with a
  95% confidence that the factor is between 1.02 and 1.04.

- insuranceyes: individuals with private insurance are expected to visit
  the physician office by a factor of 1.25, or 25% more than individuals
  without private insurance, with a 95% confidence that the factor is
  between 1.16 and 1.35.

We can see that compared to the ZINB count IRR, most of the covariate
point estimates of the IRR and the 95% CIs are very similar for all of
the variables with all the directions of the variables being in
agreement. But, the insurance and chronic variable have smaller point
estimates compared to NB, because ZINB moves a part of their effect into
the zero part.

In this dataset’s context, odds ratio means the odds of someone who
would never visit a physician’s office, which is a structural zero.
Looking at [Table 6](#tbl-or), we can see that the odds ratios for
chronic, gendermale, school, and insuranceyes are significant because
they have a p-value less than 0.05 and the CI doesn’t include 1. All of
these covariates except for gendermale have lower odds of being in the
group with a lower number of physician office visits because the odds
ratio is less than 1. Gendermale has an odds ratio over 1, meaning that
it has a higher odds being in the group with a lower number of visits to
the physician office visits compared to females.

Based on the two tables, [Table 5](#tbl-irr) and [Table 6](#tbl-or), it
is interesting to also note that for chronic and private insurance,
among people who visit the physician’s office, each additional coniditon
or having private insurance means more visits, and for people that don’t
visit a physician, both make them less likely to not have ever visited
the physician’s office. But, the opposite is true for gendermale, in
that men are more likely to have never visited the physician’s office
compared to females. However, among those who visit the physicians
office, men are less likely to make fewer visits than women.

Overall, I would prefer the ZINB model because it has the lowest AIC and
BIC compared to the Poisson and NB models and it fits the zeros the
best. ZINB model is also able to handle both pieces of information -
whether people see a physician at all and how often they go.

# 7 Discussion

The nbzinbKoo R package is able to successfully implement the negative
binomial and zero-inflated negative binomial regression from scratch.
Comparing NB to `glm.nb()`, the differences are about $`10^{-8}`$ or
smaller, well inside the $`10^{-5}`$ requirement. Comparing ZINB to
`zeroinfl()`, the largest coefficient difference is
$`1.5 \times 10^{-4}`$ on bioChemists, well inside the $`10^{-3}`$
requirement. Across 200 simulated datasets, parameter estimates show
small biases, with the largest absolute bias approximately 0.05 for the
zero-part intercept. Mean estimated standard errors are close to the
Monte Carlo standard deviations, and empirical coverage of the 95%
confidence intervals ranges from 0.930 to 0.965. These findings support
the implementation’s accuracy under the conditions examined.

There are several limitations to address for this package. First, using
the EM algorithm slows down the calculation time compared to the pscl
package’s `zeroinfl()`. For example, in ZINB on bioChemists, the
nbzinbKoo package takes 0.71 seconds vs. 0.03 seconds for `zeroinfl()`.
Additionally, for the NMES1988 dataset, it takes 1.99 seconds vs. 0.13
seconds for `zeroinfl()` ([Table 1](#tbl-agree)). Each maximization step
itself is an iterative procedure, and EM converges linearly, which is
why it takes longer. One possible extension is to use EM for the initial
iterations and then switch to Newton–Raphson updates on the full
observed-data log-likelihood. In addition, convergence does not
guarantee that the global maximum has been found. Comparing fits from
multiple starting values would provide a stronger assessment of
optimization reliability.

Another limitation is that the simulation scope is limited to one
design: $`n = 500`$, one set of true values, and $`\theta = 1.5`$. Cases
such as small n, very large $`\theta`$ (close to Poisson) and heavy zero
inflation were not tested. Future simulations could vary these
conditions to assess estimation accuracy, interval coverage, and
convergence more broadly.

Finally, the package provides model-based standard errors but does not
implement robust sandwich standard errors. Its uncertainty estimates
therefore depend on the assumed model being appropriate. Adding robust
covariance estimation would help address some forms of model
misspecification, although it would not correct misspecification of the
conditional mean.

# 8 AI-use statement

AI Statement is included in a separate pdf file in the Canvas
submission.

# 9 References

- Cameron, A. C. and Trivedi, P. K. (2013). *Regression Analysis of
  Count Data* (2nd ed.). Cambridge University Press.
- Deb, P. and Trivedi, P. K. (1997). Demand for medical care by the
  elderly: A finite mixture approach. *Journal of Applied Econometrics*,
  12(3), 313–336.
- Dempster, A. P., Laird, N. M. and Rubin, D. B. (1977). Maximum
  likelihood from incomplete data via the EM algorithm. *JRSS-B*, 39(1),
  1–38.
- Lambert, D. (1992). Zero-inflated Poisson regression, with an
  application to defects in manufacturing. *Technometrics*, 34(1), 1–14.
- Long, J. S. (1997). *Regression Models for Categorical and Limited
  Dependent Variables*. Sage.
- McCullagh, P. and Nelder, J. A. (1989). *Generalized Linear Models*
  (2nd ed.). Chapman & Hall.
- Venables, W. N. and Ripley, B. D. (2002). *Modern Applied Statistics
  with S* (4th ed.). Springer.
- Wickham, H. and Bryan, J. (2023). *R Packages* (2nd ed.). O’Reilly.
- Zeileis, A., Kleiber, C. and Jackman, S. (2008). Regression models for
  count data in R. *Journal of Statistical Software*, 27(8), 1–25.

# Appendix: side-by-side benchmark estimates

| Term        |     nbreg |    glm.nb | SE nbreg | SE glm.nb |
|:------------|----------:|----------:|---------:|----------:|
| (Intercept) |  0.256144 |  0.256144 | 0.137348 |  0.137348 |
| femWomen    | -0.216418 | -0.216418 | 0.072636 |  0.072636 |
| marMarried  |  0.150489 |  0.150489 | 0.082097 |  0.082097 |
| kid5        | -0.176415 | -0.176415 | 0.052813 |  0.052813 |
| phd         |  0.015271 |  0.015271 | 0.035873 |  0.035873 |
| ment        |  0.029082 |  0.029082 | 0.003214 |  0.003214 |
| theta       |  2.264388 |  2.264388 | 0.271176 |  0.271175 |

Table 7: NB estimates and standard errors for bioChemists: nbreg()
versus glm.nb().

| Term            |     nbreg |    glm.nb | SE nbreg | SE glm.nb |
|:----------------|----------:|----------:|---------:|----------:|
| (Intercept)     |  0.929257 |  0.929257 | 0.054591 |  0.054591 |
| hospital        |  0.217772 |  0.217772 | 0.020176 |  0.020176 |
| healthpoor      |  0.305013 |  0.305013 | 0.048511 |  0.048511 |
| healthexcellent | -0.341807 | -0.341807 | 0.060924 |  0.060924 |
| chronic         |  0.174916 |  0.174916 | 0.012092 |  0.012092 |
| gendermale      | -0.126488 | -0.126488 | 0.031216 |  0.031216 |
| school          |  0.026815 |  0.026815 | 0.004394 |  0.004394 |
| insuranceyes    |  0.224402 |  0.224402 | 0.039464 |  0.039464 |
| theta           |  1.206603 |  1.206604 | 0.033555 |  0.033555 |

Table 8: NB estimates and standard errors for NMES1988: nbreg() versus
glm.nb().

| Term               |  zinbreg | zeroinfl | SE zinbreg | SE zeroinfl |
|:-------------------|---------:|---------:|-----------:|------------:|
| count\_(Intercept) |  0.41675 |  0.41675 |    0.14360 |     0.14360 |
| count_femWomen     | -0.19550 | -0.19551 |    0.07559 |     0.07559 |
| count_marMarried   |  0.09758 |  0.09758 |    0.08445 |     0.08445 |
| count_kid5         | -0.15173 | -0.15173 |    0.05421 |     0.05421 |
| count_phd          | -0.00070 | -0.00070 |    0.03627 |     0.03627 |
| count_ment         |  0.02479 |  0.02479 |    0.00349 |     0.00349 |
| zero\_(Intercept)  | -0.19184 | -0.19169 |    1.32272 |     1.32282 |
| zero_femWomen      |  0.63595 |  0.63593 |    0.84885 |     0.84892 |
| zero_marMarried    | -1.49932 | -1.49947 |    0.93859 |     0.93867 |
| zero_kid5          |  0.62841 |  0.62843 |    0.44276 |     0.44278 |
| zero_phd           | -0.03768 | -0.03771 |    0.30799 |     0.30801 |
| zero_ment          | -0.88222 | -0.88229 |    0.31619 |     0.31623 |
| log(theta)         |  0.97637 |  0.97636 |    0.13547 |     0.13547 |

Table 9: ZINB estimates and standard errors for bioChemists: zinbreg()
versus zeroinfl().
