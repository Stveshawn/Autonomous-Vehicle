# Brake Analysis

Brake analysis based on beijing_chaoyang/arts_16_2 data from March 2019 to May 2019. The goal of this analysis is to monitor the performance of vehicle brakes:

+ Extract brake samples from the whole data stream obtained from hadoop and get some insights from the data's distribution.

+ Identify potential malfunction of the vehicle by outlier detection (t-test based on the distribution of mean acceleration in brake process)

+ Check if there is any kind of time-dependent pattern (time decay) of vehicle's performance over this time period.



`Brake_demo.py` shows a brief demostration of applying this __t-test based outlier detection__, which refers to `data_pre.py`(prepares data from hadoop raw data) and `Brake.py`(defines the class for our method),
while the notebook file contains a more complete version of the whole analysis.

More materials like EDA and conditional distribution estimation are included in the notebook file.



## Brake Data Extraction

generally we have 2 types of braking:

+ cmd change directly from positive to -57

+ cmd change from positive to 0 -> -20 -> -40 -> -57

they are actually equivalent, since the first scenario would also go through the phase (of about 300ms) to increase brake power from 0 to its maximum.

Below we describe the basic logics of identifying brake data:

+ start with `cmd==-57` and `v>eps_move`

+ end if `v<eps_still`

+ filter out invalid samples: 1. `acc==0`; 2. not in the same bag; 3. in manual driving mode;




## Outlier Detection

For outlier detection, we base it on a naive assumption that, given the brake command (`cmd==-57`) the same initial speed (`v`) and pitch condition, acceleration should normally distributed.
Then, we can estimate the distribution

```math
P(acc|cmd = -57, v, pitch)
```

and use 95% confidence interval (equivalent to t-test) to identify potential outliers.

__note__: as will be discussed later, initial velocity `v` has almost no effect on this distribution, so it's reduced to $`P(acc|cmd=-57, pitch)`$.

## Conditional Distributions

Some exploratory analysis about the distributions are implemented in the notebook file where I tried polynomial and kernel ridge to fit $`P(acc|v, cmd=-57)`$ and $`P(s_{total brake}| v, cmd = -57)`$.

I also tried quantile regression for $`P(s_{total brake}| v, cmd = -57)`$ (non-parametric), which would also be an alternative for outlier detection for total brake distance.

## Time Decay

At the end of the day, a series of EDA's are implemented to check whether there is a significant time-dependent decay in this three-month period covering different subjects and time windows.
Althought it turns out that there is no clear sign of time decay, there seems to be (as least for me) something in the mid of april that separate the data into 2 distributions.