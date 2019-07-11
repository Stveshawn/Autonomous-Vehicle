# Vehicle_Behavior_Monitoring

This repository contains the analyses for the vehicle's (beijing_chaoyang/arts_16_2) behavior of some fundamental aspects based on observed data from March 2019 to May 2019.

The analyses mainly consists of:

+ Brake Analysis

    + correctly identify and extract valid brake samples;
    
    + model conditional distributions of (__acc__ | __initial velocity__)/(__brake distance__ | __initial velocity__) and exam if there is time decay;
    
    + build a simple outlier detection procedure.

+ Throttle Analysis

    + correctly identify and extract valid throttle observations (steady speed)
    
    + exploratory analysis on the distributions of (__cmd__|__v__, __acc=0__)
    
    + build a outlier detection procedure.

+ Steering Analysis

    + filtering raw data and get valid steering observations
    
    + model distributions of (__front wheel steering angle|steering input__) and check for time decay
    
    + build a outlier detection procedure.


It is notable that none of these procedures or results are final, yet most of them should be carefully modified to satisfy the demand of the users.