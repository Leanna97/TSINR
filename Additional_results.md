# TSINR: Capturing Temporal Continuity via Implicit Neural Representations for Time Series Anomaly Detection

## Further Discussion

### Major experiments results
We compare our method with 11 other state-of-the-art approaches
on five multivariate datasets and one univariate dataset. As shown
in Table 2, **our method achieves superior overall performance on
these benchmark datasets**. These experimental results confirm that
TSINR, in both multivariate and univariate scenarios, effectively
captures temporal continuity and precisely identifies discontinuous
anomalies. The findings affirm the robustness of TSINR across
diverse datasets and showcase its potential for broader applications
in diverse domains. (If the plot is not automatically visible, please see the file of `./plot/experiments_major.png`.)

<p align="center">
<img src=./plot/experiments_major.png width="1000" height="500"/>
</p>




### Compared to other classical anomaly detection methods.
We compared our TSINR method with other classical anomaly detection methods, 
such as clustering-based methods (DeepSVDD and ITAD), 
the density-estimation models (LOF, MPPCACD, DAGMM), the classic methods (OCSVM, iForest), 
and the change point detection and time series segmentation methods (BOCPD, U-Time, TS-CP2).
(If the plot is not automatically visible, please see the file of `./plot/experiments_classical.png`.)

<p align="center">
<img src=./plot/experiments_classical.png width="1000" height="300"/>
</p>


### Additional Visualization for typical types of anomalies.
We follow DCdetector and validate the robustness of TSINR with the synthetic data which has univariate time series with different types of anomalies, 
including the point-wise anomaly (global point and contextual point anomalies) 
and patternwise anomalies (shapelet, seasonal, and trend anomalies). 
It can be seen that TSINR can robustly detect various anomalies from normal points with relatively high anomaly scores.
(If the plot is not automatically visible, please see the file of `./plot/anomalies_plot.png`.)

<p align="center">
<img src=./plot/anomalies_plot.png width="1000" height="350"/>
</p>


### Additional Ablation Study
we conduct ablation studies to verify the effectiveness of the proposed decomposition functions and group-based continuous function. 
The decomposition functions aims to capture the different components of signals (trend, seasonal and residual), 
and the group-based continuous function aims to capture information with high-frequency and non-periodic characteristics
in the residual part.
The following table shows that **integrating with the both part leads to best results.** 
(If the plot is not automatically visible, please see the file of `./plot/ablation_study.png`.)

<p align="center">
<img src=./plot/ablation_study.png width="1000" height="180"/>
</p>


### Efficiency Analysis
We measure the efficiency of the TSINR method, and show the results in the following table. The results indicate that
**the TSINR is pretty efficient and lightweight**. (If the plot is not automatically visible, please see the file of `./plot/efficiency.png`.)

<p align="center">
<img src=./plot/efficiency.png width="700" height="120"/>
</p>


### Algorithm for TSINR Anomaly Detection Algorithm
We present the algorithm for TSINR Anomaly Detection Algorithm as below. 
(If the plot is not automatically visible, please see the file of `./plot/algorithm.png`.)

<p align="center">
<img src=./plot/algorithm.png width="1000" height="350"/>
</p>
