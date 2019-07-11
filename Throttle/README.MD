# Throttle Analysis

Throttle analysis based on beijing_chaoyang/arts_16_2 data from March 2019 to May 2019. The goal of this analysis is to monitor the performance of vehicle throttle.
Specifically, we focus on examing the distribution of __velocity__ and __cmd__ when the vehicle is in a steady state. 

+ The data were filtered to exclude invalid observations for our analysis, and then split into batches.

+ Joint distributions of $`P_{t_i}(v, cmd|acc\approx 0)`$ are estimated for each batch.

+ Conditional distributions of $`P_{t_i}(cmd|v=v_{target}, acc\approx 0)`$ are obtained.

## Data Filtering

To obtain valid observations for the vehicle's steady state, I set a series of logics to filter out bad observations.

+ From prior knowledge, we know that for steady state, the vehicle's velocity should be nearly always within [1.3, 1.6], and cmd within [20, 30];

+ `acc==0` or `status&1 != 1` should be filtered out since they indicate either the sensor or the vehicle itself is not correctly working (or in automated state);

+ Large `Steering_status` and `v_diff` were fitered out to guarantee steady state we desired;

+ Only consecutive valid observations of more than 5s are included in the sample.

## Joint Distribution

Judging from EDA, we could reasonably make the assumption that given that the vehicle is running in the steady state as designed, velocity __v__ and command __cmd__ come from a bivariate normal distribution.

```math
(v, cmd|acc\approx 0)^T\sim \mathcal N((\mu_v,\mu_{cmd})^T, \Sigma_{v,cmd})
```

#### KL divergence

In the EDA part, I tried using KL divergence to visualize the difference between estimated distributions for 100 batches, where KL divergence for multivariate normal can be obtained via

```math
KL(p||q) = \frac{1}{2}\left[\log\frac{|\Sigma_2|}{|\Sigma_1|} - d + \text{tr} (\Sigma_2^{-1}\Sigma_1) + (\mu_2 - \mu_1)^T \Sigma_2^{-1}(\mu_2 - \mu_1)\right]
```

## Conditional Distribution

A conditional distribution can then be obtained from the joint distribution by the following formula in multivariate normal:

```math
(X_1^T, X_2^T)^T\sim N((\mu_1^T, \mu_2^T)^T, \begin{pmatrix}\Sigma_{11}&\Sigma_{12}\\ \Sigma_{21}&\Sigma_{22}\end{pmatrix})
```


```math
(X_1| X_2) \sim N(\mu_1 - \Sigma_{11}^{-1} \Sigma_{12}(x_2-\mu_2), \Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21})
```

By this conditional distribution, we could know that __for a specific target steady speed, what would be the reasonable interval for command__ and a outlier detection procedure based on confidence interval 
can then be subsequently constructed.

## Code instruction

`Throttle.py` defines the class for some basic methods of this analysis, and refer `Throttle_domo.py` for a brief demonstration. More details are included in the notebook file.