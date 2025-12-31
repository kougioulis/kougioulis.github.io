---
layout: distill 
title: "Beyond Prediction: Why Causality Matters"
giscus_comments: true
date: 2025-12-31 10:00:00+0200
description: And what your ML models are missing
tags: causality, causal-ML, citation
categories: Comments # for giscus
citation: true 
thumbnail: assets/img/bde52737.png
bibliography: 2025-12-31.bib

toc:
  - name: "Introduction"
  - name: "Why Causation Matters"
  - name: "The Common Cause Principle"
  - name: "## The Language of Causality - SCMs & Pearl's Ladder"
  - name: "How Do We Discover Causal Graphs"
  - name: "An Illustrative Example"
  - name: "Closing Thoughts"
---

### Introduction 🥚🐥

Modern data systems, from recommendation engines to climate analytics, are built almost entirely on *predictive modeling*. In a nutshell, one collects a vast number of observational samples (that are created under variable experimental scenarios), fits increasingly complex models (from simple regression models to fancy deep neural networks), and optimize for accuracy. Pretty much, classification models are fitted on target variables of interest, and decision policies are taken simply using these models. If $f$ is such a fitted classification model and $X$ a vector of features (e.g. cost of acquiring a customer and manufacturing expenses), then one may fit a predictive model on a variable of interest $y$ (e.g. expected sales) by $y = f(X)$ and then use it for sales prediction by plugging in values of $X$. Beneath the surface however lies a <b>fundamental limitation</b> of predictive models, which becomes crucial whenever we want to answer questions such as:

- ❓ What will happen if I set the price of a product to, let's say, $5$ euros? Will sales increase?

- ❓ Why did an outcome occur?

- ❓ How do we make decisions that remain valid when conditions change (e.g. we test a drug on mice and are interested whether its effect changes under a distribution shift, e.g. on a population of humans)

Overall, predictive models excel at recognizing associations. Causal models on the other hand represent the underlying mechanisms of data. The difference between the two, although subtle at first, defines the boundary between pattern recognition and scientific reasoning. As Scholkopf et al. <d-cite key="scholkopf2021towards"></d-cite> point out:

<blockquote>
If we wish to incorporate learning algorithms into human decision making, we need to trust that the predictions of the algorithm will remain valid if the experimental conditions are changed.
</blockquote>

In this post, we’ll walk through why causal models matter, how causal reasoning differs from prediction and illustrate the stakes, along with a minimal example. It does not serve as a complete treatment of causality, but as a motivational introduction to the unfamiliar reader. For a thorough treatment, we refer the interested reader to <d-cite key="pearl2009causality"></d-cite>, <d-cite key="spirtes2001causation"></d-cite>, and <d-cite key="pearl2018book"></d-cite>. Chapter 1 of <d-cite key="kougioulis2025large"></d-cite> serves as a detailed version of this blog post.

## Why Causality Matters 

Predictive models rely on observed *correlations* and *patterns* in observational data (i.e. samples that are purely observed, not obtained under a specific manipulation of the examined system). They implicitly assume that all samples come from a <b>single, stable distribution</b> (the familiar i.i.d. assumption). Under this assumption, identifying strong associations can be enough to make good and powerful predictions. 

However, associations alone cannot tell us <b>what causes what</b>. Consider the following, basic example:

### Predicting Ice Cream Sales 🍦

It is known that during the summer, ice cream sales are increased compared to other seasons, with number of sunburn cases also showing an higher trend. We know (taken as expert knowledge) that although these two quantities are correlated, ice cream sales do not cause sunburn cases and vice-versa. Instead, these two quantities share a *confounder (common cause)* like the sun's radiation, or even temperature. In any case, even if they have more than one confounding variable, we are allowed to treat them both as a single. The fact that $\text{ice cream sales} \leftarrow sun \rightarrow \text{sunburn}$, highlights our previous discussions, where a directed arrow represents a direct causal relationship (from a cause to its direct effect(s)).

Now imagine an alternate world where everyone wears suncreen 🧴 (we intervened on sun's radiation indirectly, by forcing lower radiation by sunscreen use): Sunburn cases plummet, but ice cream sales remain unchanged.Will ice cream sales increase, decrease or remain unaffected (similarly for sunburn cases)? 

A predictive model would infer: <b>more sunburn → more ice cream sales</b>. We know this is wrong, but the model doesn’t. Both variables are effects of a <i>third, hidden cause:</i> <b>sunlight intensity</b>. This unobserved confounder creates misleading correlations, and as a result, any predictive model trained on the original correlation will <b>catastrophically fail</b>, and any inferred decisions cannot be taken seriously.

This example captures the core limitation of predictive modeling:

<blockquote>
💡 Predictive patterns break when the observed environment changes, while causal mechanisms do not.
</blockquote>

The following table briefly shows scenarios where predictive and causal queries differ, and as a result, how predictive models can lead to false interpretations.

| **Task**              | **Query**                                          | **Example**                   | **Description**                                                | **Causal Model**                          | **Predictive Model**                               |
|-----------------------|----------------------------------------------------|-----------------------------------------|----------------------------------------------------------------|--------------------------------------------|------------------------------------------------------|
| **Prediction**        | Predict / diagnose **$Y$ given $X$**                  | “What is $Y$ when $X_1$ = 5?”                | Standard supervised prediction                                 | ✔️ Correct predictions                     | ✔️ Correct predictions                              |
| **Decision Making**   | Optimal $X$ to increase $Y$ under constraints | “What $X_1$ maximizes Y given $X_2 = 6$?”     | Choosing an action that changes the system                     | ✔️ Correct decisions                       | ❌ Possibly wrong decisions                         |
| **What-if**           | Hypothetical changes (interventions)              | “What if I set $X_1 = 5$?”                 | Interventional reasoning, requires *$do(X)$*                     | ✔️ Correct estimate                        | ❌ Possibly wrong estimate                          |
| **Interpretation**    | Feature importance / effect of $X$ on $Y$             | “Does $X_1$ affect $Y$?”                     | Understanding influence of features                             | ✔️ Correct estimate                        | ❌ SHAP/feature importance may be misleading        |
| **Counterfactual**    | “What would $Y$ have been if $X$ had been different?” | “$Y=3$ when $X_3=$yellow. What if $X_3$=green?” | Individual-level alternative-world reasoning                    | ✔️ Correct estimate                        | ❌ Generally impossible                            |
| **Root Cause**        | Identify cause of an event                         | “What caused the failure?”              | Find initial causal driver                                      | ✔️ Correct estimate                        | ❌ Possibly wrong                                    |

## The Common Cause Principle

But how can observational data mislead us? Reichenbach’s <b>common cause principle</b><d-cite key="reichenbach1956direction"></d-cite> states:

> 💡 If X and Y are correlated, then either X causes Y, Y causes X, or a hidden confounder causes both.

Observational data cannot tell these apart. This limitation explains famous phenomena like <b>Simpson’s paradox</b><d-cite key="simpson1951interpretation"></d-cite>, where aggregated correlations reverse once you account for confounders. It also explains why <b>SHAP values and feature importance</b>, though useful, are not causal measures. They reflect importance <b>within the model</b>, not influence <b>in the real world</b>.


## Randomized Experiments: The Gold Standard for Causality

Given the limitations of observational data, a natural question arises: <i>how can causal effects be measured correctly?</i> Since the seminal work of Ronald Fisher<d-cite key"fisher1935design" />, the gold standard for causal inference has been <b>randomized experimentation</b>, and in particular <b>Randomized Controlled Trials (RCTs)</b>.

Randomized experiments aim to isolate causal effects by deliberately intervening on one or more variables of interest while holding all other factors constant <b>in expectation</b>. This is typically achieved through <b>random assignment</b>, which ensures that both observed and unobserved covariates are, on average, balanced across experimental groups. As a result, randomization removes confounding <i>by design</i>, allowing causal effects to be identified without relying on strong modeling assumptions.

<div class="row justify-content-sm-center">
  <div class="col-sm-auto text-center mt-4 mt-md-0">
    {% include figure.liquid path="assets/img/ab-testing.png" class="img-fluid rounded z-depth-1" style="max-width: 80%; height: auto;" %}
    <div class="caption mt-2" style="text-align: center;">An A/B test on an online platforms is an instance of a randomized controlled trial (RCT). Users are randomly assigned to interact with one of <i>two</i> webpage versions, for example, a new interface (left) or the existing <i>control</i> version (right). Outcomes of interest (e.g., user engagement) are measured over a fixed time period. Statistical analysis is then used to determine whether the introduced interface has a statistically significant causal effect on the outcome variables (e.g., exposure rates or sales). Illustration courtesy of Abstraktmg<d-footnote>https://www.abstraktmg.com/a-b-testing-in-marketing/
 (last accessed: 30 December 2025).</d-footnote>.
    </div>
  </div>
</div>

Common instances of randomized experimentation include:

* <b>Randomized Controlled Trials (RCTs)</b> in medicine and the social sciences,
* <b>A/B tests</b> in online platforms and digital systems,
* <b>Controlled laboratory experiments</b> in the natural sciences,
* <b>Simulation-based interventions</b> in synthetic or virtual environments.

To illustrate, consider a clinical trial investigating whether a newly introduced treatment (let's call it $X$) reduces the severity of migraines. An RCT proceeds by randomly assigning patients to either a <b>treatment group</b> (receiving treatment X) or a <b>control group</b> (receiving a placebo), and subsequently comparing outcomes across the two groups. Because assignment is random, any systematic difference in outcomes can be attributed to the intervention itself rather than to pre-existing differences among patients.

A standard causal quantity estimated in such settings is the <b>Average Treatment Effect (ATE)</b>, defined as $\text{ATE} = \mathbb{E}[Y_1 - Y_0]$ where $Y_1$ and $Y_0$ denote the potential outcomes under treatment and control, respectively. If the ATE is statistically distinguishable from zero, one concludes that the intervention has a causal effect on the outcome of interest.

The conceptual strength of RCTs lies in their <b>ability to eliminate confounding bias through randomization</b>, making them the most reliable tool for causal inference. However, despite their methodological appeal, randomized experiments are often <b>impractical or infeasible</b> in real-world settings. They can be prohibitively expensive, logistically complex, or ethically unacceptable, for instance, when studying the causal effects of harmful behaviors such as smoking. Moreover, in modern large-scale systems, the space of possible interventions is often combinatorial, rendering exhaustive experimentation impossible in practice.

These limitations motivate the development of <b>model-based causal frameworks</b> that allow causal effects to be inferred without relying exclusively on randomized experiments. In the following sections, we introduce such a framework through structural causal models, which enable principled reasoning about interventions, counterfactuals, and distributional changes beyond purely observational data.

## The Language of Causality - SCMs & Pearl's Ladder

The formal language in Causality to reason about causal mechanisms, interventions and counterfactuals are <b>Structural causal models (SCMs)</b>, introduced by Judea Pearl <d-cite key="pearl2009causality"></d-cite>. An SCM consists of:

1. <b>A Directed Acyclic Graph (DAG)</b> where nodes represent the examined variables of interest and direct arrows <b>direct causal effects</b> (e.g., Sun → IceCream).
2. <b>Structural equations</b> ($X_i = f_i(\mathrm{Parents}(X_i), \epsilon_{X_i})$), representing how each variable $X_i$ is generated.
3. <b>Independent noise terms</b> $\epsilon_{X_i}$ for each variable $X_i$, representing unobserved and inherent randomness of the system.

This framework allows us to compute three fundamentally different types of queries:

| Type               | Question                                 | Example                  |
| ------------------ | ---------------------------------------- | ------------------------ |
| <b>Observational</b>  | What do we see?                          | $(P(Y \mid X=x))$          |
| <b>Interventional</b> | What happens if we <i>force</i> X to a value? | $(P(Y \mid do(X=x)))$      |
| <b>Counterfactual</b> | What would have happened <i>otherwise</i>?    | $(P(Y_{x'} \mid X=x,Y=y))$ |

<aside><p>These three different types of queries are known as <b>Pearl's Ladder of Causal Hierarchy</b> <d-cite key="pearl2018book"></d-cite>, where predictive models can only answer queries of level 0, with causal models additionally to level 1 and 2.</p> </aside>

Predictive models only answer the first type, while causality accounts for all three.

SCMs allow us to simulate interventions mathematically without performing physical experiments. Using <b>do-calculus</b>, we can compute $P(Y \mid do(X=x))$ which differs from the purely observational $P(Y \mid X=x)$ except in special, unconfounded cases (outside the scope of this blog). This gives us a way to predict interventions, <i>if we know the causal graph</i>.

## How Do We Discover Causal Graphs?

This is where the field of <b>causal discovery</b> enters. Causal discovery attempts to learn the structure of the system (the DAG) from data (often observational, sometimes including interventional samples), which operate under certain causal assumptions. For example, the causal sufficiency assumption assumes no hidden confounders, while faithfulness (loosely defined in the scope of this blog post) that no determinism exists in the examined system, hence no causality (e.g. variables that are deterministically related like ratios of variables). Once the graph is known, <b>causal inference</b> methods estimate effects such as:

* ATEs
* mediation effects
* optimal intervention strategies
* counterfactual outcomes

In practice, as the ground truth causal model is unknown, one first discovers the causal structure (the causal DAG and the SCM representation) and then estimates causal effects.

## An Illustrative Example <d-footnote>The provided examples are available in a curated Github Repository (https://github.com/kougioulis/causality-tutorial) with various causality tutorials for the motivated reader.</d-footnote>

Consider 3 variables, $X,Y,Z$. $X$ and $Z$ are observed to have correlation coefficient $3$. That is, if you observe $1$ unit of change in $X$, you observe $3$ units of change in $Z$. Let's try to answer the following questions:

<aside> <p>For simplicity purposes, we assume linear relationships.</p> </aside>

- ❓ If you intervene and change $X$ (not merely observe) in the real world, what change would you observe?
- ❓ If you build a predictive model from $X$ to $Z$ and change the input to the model, would you observe the same change when you intervene in the real world?

For linear correlations and causal relations with Gaussian additive noise, we make the following assumptions:
- ☝️ Correlations on a path <b>multiply</b> together
- ☝️ Correlations from different path <b>sum</b>

We begin by importing some needed modules:

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

np.random.seed(42)
```

Consider a <b>causal</b> DAG, assuming linear relationships with edges $X \leftarrow Y \rightarrow Z$ and $X \rightarrow Z$. A directed edge illustrates direct causal influence (i.e. $X$ directly causes $Y$, $X$ is the direct cause of $Y$, $Y$ is the direct effect of $X$) while indirect causal relationships are illustrated by directed paths. The linear coefficients of each causal edge are $0.7$ for $Y \rightarrow X$, $-8$ for $Y \rightarrow Z$ and $13$ for $X \rightarrow Z$. Loosely defined, pairing this causal DAG with the functional dependencies of its variable given its parents (direct causes), plus an additive noise term (accounting for randomness), creates an (additive) structural causal model (SCM). Let's try to answer the following question:

<blockquote>What is going to happen when I increase X by 1?</blockquote> 

- The non-causal path $X \leftarrow Y \rightarrow Z$ has coefficient $0.7 \cdot (-8)=-5.6$. 
- The path $X -> Z$ coefficient $13$ causal effect causal path.
- The total correlation coefficient observed is $0.7 - 8 = -7.3$.

We observe that the *causal effect is larger than the computed correlation!* (Causal effect is positive, observed correlation is negative)

We define the SCM of the three variables:

```python
# Defining structural equations of the SCM
# X <- Y (coef 5) + noise
# Z <- Y (coef -2) + X (coef 13) + noise

n_samples = 10**6  # large enough sample size

# Exogenous Gaussian noise
eps_y = np.random.normal(0, 1, n_samples)
eps_x = np.random.normal(0, 1, n_samples)
eps_z = np.random.normal(0, 1, n_samples)

# Structural Causal Model
Y = eps_y
X = 0.7 * Y + eps_x
Z = -8 * Y + 13 * X + eps_z

data = pd.DataFrame({"X": X, "Y": Y, "Z": Z})
```

And compute the observed Pearson correlation between X and Z, as well as the predictive effect (regression $Z \sim X$):

```python
# Pearson Correlation between X and Z (observational / predictive)
corr_xz = np.corrcoef(X, Z)[0, 1]

# Predictive model: Z ~ X by ordinary least squares regression
X_df = sm.add_constant(data["X"])
model_pred = sm.OLS(data["Z"], X_df).fit()
pred_effect = model_pred.params["X"]

print("Observed correlation between X and Z:", round(corr_xz, 3)) # 0.862
print("Predictive effect (regression Z ~ X):", round(pred_effect, 3)) # 9.246
```
From the above snippet, we obtain the following contradictory results:

- <b>Causal model:</b> Increase $X$ by $1$, increase Z by $0.86$.

- <b>Predictive model:</b> Increase $X$ by $1$, increase Z by $9.25$. 

The correct approach, would instead be to:

- 1️⃣ Learn the causal model (via a <b>causal discovery</b> algorithm, in this example we assume it is known correctly.)
- 2️⃣ Identify the <b>non-causal paths</b> and remove the effect of the non-causal paths and only (identify the quantities that block the correlations from non-causal paths, called the <b>adjustment set</b>).

<aside> <p><b>Adjustment set:</b>a set of variables that, when conditioned on, blocks all backdoor (confounding) paths between a treatment and an outcome, enabling unbiased causal effect estimation from observational data.</p> </aside>
- 3️⃣ Build a <b>predictive model</b> that <b>includes an adjustment set</b> and only and hence controls for their values. 

The correlation coefficient of $X$ to $Z$ conditioned on fixed values of $Y$ provide the <b>true causal effect</b> of $13$ units with the following code snippet, with an absolute error in the third decimal place.

```python
# True causal effect (by an intervention do(X))
causal_effect = 13  # known from our structural equations

# Adjustment set approach (control for Y in regression)
XZ_df = sm.add_constant(data[["X", "Y"]])
model_adj = sm.OLS(data["Z"], XZ_df).fit()
adj_effect = model_adj.params["X"]

print(f"Causal effect (true do-intervention): {causal_effect}") # 13
print(f"Causal effect via adjustment set (regression Z ~ X + Y): {round(adj_effect, 3)}") # 13.002
```

<div class="row justify-content-sm-center">
  <div class="col-sm-auto text-center mt-4 mt-md-0">
    {% include figure.liquid path="assets/img/interv_obs.png" class="img-fluid rounded z-depth-1" style="max-width: 80%; height: auto;" %}
    <div class="caption mt-2" style="text-align: center;">Histogram of the conditional observational distribution $Z|X=1$ against the interventional distribution $Z|\text{do}(X=1)$ further illustrates the difference between the two (and convince us that in general, $P(Z | X=x) \neq P(Z | do(X=x))$).</div>
  </div>
</div>

As shown in the histogram, the two distributions $Z \mid X \approx 1$ and $Z \mid do(X = 1)$ differ dramatically because:
 
- Under <b>conditioning</b>, when we observe $X = 1$, that typically means <b>Y was high</b>, because $X = 0.7Y + \text{noise}$.
  Thus the backdoor path $X \leftarrow Y \rightarrow Z$ pushes Z down via the $-8Y$ term.
 
- Under <b>intervention</b>, when we force $X = 1$, $Y$ is no longer correlated, so only the direct causal effect $(13X)$ applies.

## Back to the Ice Cream Sales Example

Let's return to the ice cream example illustrated at the beginning of the post, but this time with some code:


```python
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
```

```python
samples = 500  # observational samples (iid)

# True causal mechanism: ice_cream sales <- sun -> sunburn
sun = np.random.uniform(0, 10, size=samples)
ice_cream = 2 * sun + np.random.normal(0, 1, size=samples)
sunburn = 3 * sun + np.random.normal(0, 1, size=samples)

# Predictive model incorrectly learns sunburn -> ice cream
X = sunburn.reshape(-1, 1)
pred_model = LinearRegression().fit(X, ice_cream)

# Intervene: sunscreen campaign sets sunburn to a constant
sunburn_intervened = 3 * np.ones(samples)

# Predictive model's WRONG prediction under intervention
predicted_icecream = pred_model.predict(sunburn_intervened.reshape(-1, 1))

# True causal outcome under intervention - ice cream depends ONLY on sunlight, not sunburn (graph surgery is performed on the underlying SCM)
true_icecream = 2 * sun + np.random.normal(0, 1, size=samples)

print(f'True Ice Cream Sales (causal model): {true_icecream.mean():.2f} units') # 10.04 units
print(f'Predicted Ice Cream Sales (predictive model): {predicted_icecream.mean():.2f} units') # 2.05 units 
```

A predictive model wrongly suggests that an increase in one unit of radiation will account for two units of increase in ice cream sales, which is 5 times less that the true causal effect which we obtained using the true causal model.

As we have noticed already, a predictive model is doomed to failure if applied to cases outside the trained distribution: For example, consider applying the above simple regression model on a hypothetical subpopulation of Scandinavians (who are less exposed to sunlight compared to other regions) who enjoy eating ice cream all year long. 

## Closing Thoughts

Predictive models excel at finding patterns in data, but patterns alone are not enough when decisions, interventions, or changing environments are involved. Predictive models (either from classical ML up to deep learning approaches) have found great sucess, yet they prove unstable when the underlying perturbed.

<div style="max-width: 450px; margin: 0 auto; zoomable:true">
  {% include figure.liquid 
     path="assets/img/causality_meme.jpeg" 
     class="img-fluid rounded z-depth-1" %}
</div>

Causal modeling addresses this gap by explicitly representing <b>how</b> data is generated. By reasoning about interventions and counterfactuals, causal methods allow models to generalize beyond the conditions under which they were trained and to support meaningful actions.

It is to no surpise that decades after Pearl's formulation of causality, the industry is just starting to adopt causal discovery and causal inference methods for optimized decision making and creation of <i>causal digital twins</i>, especially in the case of time-series data (by Temporal Structural Equation Models - TSCMs<d-cite key="runge2018causal" /><d-cite key="runge2019inferring" />), now called <b>Causal AI</b> to contrast approaches using traditional ML or deep learning and LLMs.

<aside><p> A causal digital twin is a causal model of an examined system that combines a real-time state representation alongside a causal model (e.g., a structural causal model), allowing counterfactual reasoning, intervention analysis, and policy evaluation. </p></aside>

What's your view? Let me know in the comments! 🚀