# Steering Analysis

Steering analysis based on beijing_chaoyang/arts_16_2 data from March 2019 to May 2019. The goal of this analysis is to monitor the vehicle's steering performance by modeling the
distribution of front wheel steering angle $`\delta`$ given the `steering_status` $`x`$, $`P(\delta|x)`$:

+ Filter out invalid observations which should not be included in the model we build, such as observations with:
    
    1. Negative command; 2. small velocity; 3. acceleration being exactly 0; 4. manual operation; 5. very large steering angle.

+ Seperate the filtered data into smaller batches and build a model of $`P_{t_i}(\delta| x)`$ for each of them, and exam whether there is any time-dependent pattern for the distributions.

+ Build a Bayesin Linear Regression Model for outlier detection.


## Data Filtering

As mentioned earlier, I filtered out some invalid observations which should not be included in our final model.

1. Negative command: since we are looking at the steering performance, we should focus on frames whose `cmd` is not negative;
2. small velocity: same as the previous one;
3. acceleration: acceleration being exactly 0 means that the sensor is not working correctly;
4. manual operation: we focus on the performance of automated driving;
5. large steering angle: there are cases where large steering statuses are caused by incorrect sensoring at the very beginning of each trip.
6. bags which has too much manual operation are removed since this might suggest the vehicle is not in the correct condition as expected.


## Model Distribution

we can choose to either seperate the data (chronologically sorted) into evenly-spaced batches or seperate it by different bags.
And On each batch, we choose to build a linear regression to model the distribution $`P_{t_i}(\delta|x)`$.

Methods including EDA, hypothesis testing (stationary test, Hotelling's T-squared test) and GMM are applied to exam the change of distribution $`P_{t_i}(\delta|x)`$ over time (see `Steering Analysis.ipynb` for details).

## Outlier Detection

For outlier detection, we resort to Bayesian Linear Regression and it's posterior credible intervals/region.

+ Basic scheme (OLS)

```math
\delta = ax+b+\epsilon,\qquad \epsilon \sim N(0, \sigma^2)
```

where $`\delta`$ is the __front wheel steering angle__ (derived from formula $`\delta = \arctan(\frac{\omega_{yaw}L}{v_{x}})`$), $`x`$ is __steering_status__, $`\omega_{yaw}`$ is the angular velocity and $`v_x`$ is __velo_robot_x__

+ Bayesian Linear Regression (BLR) scheme

```math
\delta = ax+b+\epsilon,\qquad \epsilon \sim N(0, \sigma^2),\ a\sim N(\mu_a, \sigma^2_a),\ b\sim N(\mu_b, \sigma^2_b)
```

use the distribution of $`a, b`$ to identify outliers of coefficients and intercepts.

__Or__ using another parametrization, we have

```math
\delta = \tilde X\beta+\epsilon
```

where $`\tilde X=[\mathbf 1, x]`$, $`\beta = (b, a)^T`$, $`\epsilon\sim N(0,\sigma^2)`$, with conjugate priors (normal-gamma conjugate scheme)

```math
\beta\sim N(\mu_0, \Lambda_0^{-1}),\qquad \phi=(\sigma^2)^{-1}\sim Gamma(a_0,b_0)
```


Then the posterior distribution of parameters given observed data can de induced by

```math
p(\beta, \phi| X, y) \propto p(\beta, \phi)p(y|\beta, \phi, X)\propto p(\beta| \phi)p(\phi)p(y|\beta, \phi, X)
```

The parameters for posterior distribution are as follows:

```math
\Lambda_n = X^TX+ \Lambda_0\qquad
\mu_n = \Lambda_n^{-1}(X^TX\hat\beta_{ols}+\Lambda_0\mu_0)\qquad
a_n = a_0+\frac{n}{2}\qquad
b_n = b_0+\frac{y^Ty+\mu_0^T\Lambda_0\mu_0-\mu_n^T\Lambda_n\mu_n}{2}
```


## Code instruction

run `Steering_demo.py` to show a brief demonstration of data filtering, distribution estimating and outlier detection.

More details and analyses might be found in the notebook file.