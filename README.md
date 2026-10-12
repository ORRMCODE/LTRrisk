# LTRrisk: Risk-Aware Learning to Rank

A portfolio-theoretic framework for training ranking models that balance average effectiveness with robustness to uncertain relevance.

Based on the paper **“A General Framework for Risk Optimization with Learning to Rank through Portfolio Theory.”**

## Overview

Standard learning-to-rank (LTR) objectives typically maximize expected ranking utility. When relevance is uncertain, a high average score can hide a substantial probability of poor outcomes. For example, when a ranking serves only one interpretation of an ambiguous query.

This work incorporates risk functionals from portfolio theory directly into **exposure-based LTR**. Rather than adding a greedy re-ranking step at inference time, the framework optimizes a stochastic ranking model using samples from a relevance distribution.

The portfolio analogy is straightforward:

| Portfolio theory | Learning to rank |
| --- | --- |
| Assets | Documents |
| Investment allocation | Expected document exposure |
| Uncertain asset returns | Uncertain document relevance |
| Portfolio utility | Exposure-weighted relevance |

Risk-aware optimization can encourage diversification without an explicit diversity term in the objective.

## Main Contributions

- A general framework connecting portfolio-theoretic risk optimization with ranking-model training.
- Sample-based estimators for seven risk functionals.
- Integration with exposure-based gradient estimation and automatic differentiation.
- Experiments examining both independent and correlated relevance uncertainty.

## Optimization Objective

For a query, the utility of a ranking policy is expressed through its expected document exposures:

$$
U(\pi \mid q,\rho)=\sum_d \theta_{d\mid\pi,q}\rho_{d\mid q}.
$$

The **risk-corrected expected utility** combines mean utility and a risk-sensitive functional:

$$
\mathcal{U}_{\lambda}(\pi\mid q)
=(1-\lambda)\mathbb{E}_{\rho}[U(\pi\mid q,\rho)]
+\lambda\mathcal{R}_{\rho}[U(\pi\mid q,\rho)].
$$

Here, $\pi$ is the stochastic ranking policy, $\theta_{d\mid\pi,q}$ is expected document exposure, and $\rho$ is sampled relevance. The parameter $\lambda\in[0,1]$ controls the trade-off between average utility and the risk-sensitive objective.

**Important:** risk is evaluated over relevance uncertainty after averaging over the policy’s rankings. The objective does not directly control the outcome of each individual sampled ranking.

### Training Procedure

1. Estimate expected document exposure by sampling rankings from the policy.
2. Sample relevance vectors from an uncertainty model.
3. Compute exposure-weighted utility for each relevance sample.
4. Estimate mean utility and the selected risk functional.
5. Combine them into the risk-corrected objective.
6. Update the ranking model through exposure-based gradient estimation.

The relevance model only needs to support sampling; the framework does not require an explicit probability density. Relevance samples are treated as fixed inputs during ranking-model optimization.

## Risk Functionals

| Functional | Optimization target |
| --- | --- |
| Variance | Reduce overall utility dispersion |
| Standard deviation | Reduce dispersion on the same scale as utility |
| Lower semivariance | Reduce squared deviations below mean utility |
| Lower semi-standard deviation | Reduce downside dispersion on the utility scale |
| Value at Risk (VaR) | Improve a selected lower utility quantile |
| Average Value at Risk (AVaR / CVaR) | Improve average utility in the worst-performing fraction of outcomes |
| Entropic Value at Risk (EVaR) | Optimize a conservative lower-tail bound based on exponential moments |

Variance-based penalties enter the maximization objective with a negative sign. VaR, AVaR, and EVaR use a **utility convention**: higher values are better. Their parameter $\alpha\in(0,1)$ specifies the lower-tail level, with smaller values targeting more extreme downside outcomes.

## Experimental Setup

Experiments use **MSLR-Web-10K** and **Istella-S**, with **DCG@10** as the utility metric.

Relevance uncertainty is simulated using label-frequency distributions obtained from **25 K-means document clusters**:

- **Per-document sampling:** each document independently receives a label sampled from its cluster’s distribution.
- **Per-cluster sampling:** one label is sampled for each cluster and shared by its documents, creating correlated relevance outcomes.

### Reported Training Configuration

| Setting | Value |
| --- | --- |
| Hidden-layer widths | 256, 128, 64 |
| Training epochs | 500 |
| Optimizer | AdamaxW |
| Learning rate | 0.01 |
| Query batch size | 2,750 |
| Ranking samples per exposure estimate | 120 |
| Relevance samples per utility estimate | 1,000 |
| Independent runs | 10 |

The paper describes a JAX-based implementation using exposure-based LTR.

### Evaluation

Models are compared with an expected-utility baseline and an expected-utility-plus-exposure-entropy baseline. Evaluation reports:

- Mean DCG@10.
- Median, 25th-percentile, and 10th-percentile utility.
- Exposure entropy.
- The fraction of documents needed to account for 95% of exposure.

## Main Findings

- **Optimizing the mean alone can produce poor lower-tail outcomes**, particularly under correlated relevance uncertainty.
- **Risk-aware objectives can improve downside performance**, sometimes at the cost of average utility.
- **Diversification emerges naturally**, although higher diversity is not itself a guarantee of lower risk.
- **Semi-standard deviation and VaR-based objectives generally provide more favorable trade-offs than raw variance** in the reported experiments.
- **No single risk functional is universally best**; the appropriate choice depends on the uncertainty structure and desired trade-off.

## Scope and Limitations

- Experimental relevance distributions are **simulated**, not directly observed uncertainty annotations.
- Results depend on the relevance uncertainty model, including correlations between documents.
- The framework optimizes policy-level exposure utility rather than risk for individual rankings.
- Simple sample-based estimators prioritize practical optimization; their automatically differentiated gradients are not guaranteed to be unbiased or consistent.
- Strong risk aversion can substantially reduce mean effectiveness.