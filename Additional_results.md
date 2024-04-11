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
in diverse domains

<p align="center">
<img src=./plot/experiments_major.png width="1000" height="500"/>
</p>




### Compared to other classical anomaly detection methods.
We compared our TSINR method with other classical anomaly detection methods, 
such as clustering-based methods (DeepSVDD and ITAD), 
the density-estimation models (LOF, MPPCACD, DAGMM), the classic methods (OCSVM, iForest), 
and the change point detection and time series segmentation methods (BOCPD, U-Time, TS-CP2).

<p align="center">
<img src=./plot/experiments_classical.png width="1000" height="300"/>
</p>


### Additional Visualization for typical anomalies.
We use the synthetic data from DCdetector, which has univariate time series with different types of anomalies, 
including the point-wise anomaly (global point and contextual point anomalies) 
and patternwise anomalies (shapelet, seasonal, and trend anomalies). 
It can be seen that TSINR can robustly detect various anomalies better from normal points with relatively higher anomaly scores.

<p align="center">
<img src=./plot/anomalies_plot.png width="1000" height="350"/>
</p>


### Additional Ablation Study
we conduct ablation studies to verify the effectiveness of the proposed decomposition functions and group-based continuous function. 
The decomposition functions aims to capture the different components of signals (trend, seasonal and residual), 
and the group-based continuous function aims to capture information with high-frequency and non-periodic characteristics
in the residual part.
The following table shows that **integrating with the both part leads to best results.** 

<p align="center">
<img src=./plot/ablation_study.png width="1000" height="180"/>
</p>