# TSINR: Capturing Temporal Continuity via Implicit Neural Representations for Time Series Anomaly Detection

<p align="center">
<img src=plot/motivation.png width="1100" height="300"/>
</p>

## Abstract
Time series anomaly detection aims to identify unusual patterns in data or deviations from systems’ expected behavior. The re-construction-based methods are the mainstream in this task, which exploit learning point-wise representation. However, the unlabeled anomaly points in training data may cause these reconstruction-based methods to learn and reconstruct anomalous data, resulting in the challenge of capturing normal patterns. In this paper, we propose a time series anomaly detection method based on implicit neural representation (INR) reconstruction, named TSINR, to address this challenge. Due to the property of spectral bias, TSINR enables prioritizing low-frequency signals and exhibiting poorer performance on high-frequency abnormal data. Specifically, we adopt INR to parameterize time series data as continuous functions and employ a transformer-based architecture to predict the INR of given data. As a result, the proposed TSINR method achieves the advantage of capturing the temporal continuity and thus is more sensitive to discontinuous anomaly data. In addition, we further design a novel continuous function to learn inter- and intra-variable information, and leverage a pre-trained large language model to amplify the intense fluctuations in anomalies. Extensive experiments demonstrate that TSINR achieves superior overall performance on both univariate and multivariate time series anomaly detection benchmark compared to other state-of-the-art reconstruction-based methods.


## Reconstruction from INR
<p align="center">
<img src=plot/architecture.png width="1100" height="500"/>
</p>


## Get Started
Please install Python>=3.8 and install the requirements via:
```
pip install -r requirements.txt
```

### Standard benchmark
Please download the benchmarks from [TimesNet](https://github.com/thuml/Time-Series-Library) and store the data in `./all_datasets`.


Then please run the TSINR method with following command by choosing the configuration from `./cfgs`:
```
CUDA_VISIBLE_DEVICES=0 python run_trainer.py --cfg <path to cfg>
```

For example:
```
CUDA_VISIBLE_DEVICES=0 python run_trainer.py --cfg ./cfgs/MSL.yaml
```

### Experiment for UCR datasets
Please download the ucr dataset from [here](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip) and store the data in `./all_datasets`.
Then please run the experiments with the following command:

```
CUDA_VISIBLE_DEVICES=0 python run_trainer_ucr.py --cfg ./cfgs/UCR.yaml
```


## Major Results
<p align="center">
<img src=plot/experiments_major.png width="1100" height="600"/>
</p>

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
<img src=plot/experiments_major.png width="1000" height="500"/>
</p>




### Compared to other classical anomaly detection methods.
We compared our TSINR method with other classical anomaly detection methods, 
such as clustering-based methods (DeepSVDD and ITAD), 
the density-estimation models (LOF, MPPCACD, DAGMM), the classic methods (OCSVM, iForest), 
and the change point detection and time series segmentation methods (BOCPD, U-Time, TS-CP2).
(If the plot is not automatically visible, please see the file of `./plot/experiments_classical.png`.)

<p align="center">
<img src=plot/experiments_classical.png width="1000" height="300"/>
</p>


### Additional Visualization for typical types of anomalies.
We follow DCdetector and validate the robustness of TSINR with the synthetic data which has univariate time series with different types of anomalies, 
including the point-wise anomaly (global point and contextual point anomalies) 
and patternwise anomalies (shapelet, seasonal, and trend anomalies). 
It can be seen that TSINR can robustly detect various anomalies from normal points with relatively high anomaly scores.
(If the plot is not automatically visible, please see the file of `./plot/anomalies_plot.png`.)

<p align="center">
<img src=plot/anomalies_plot.png width="1000" height="350"/>
</p>


### Additional Ablation Study
we conduct ablation studies to verify the effectiveness of the proposed decomposition functions and group-based continuous function. 
The decomposition functions aims to capture the different components of signals (trend, seasonal and residual), 
and the group-based continuous function aims to capture information with high-frequency and non-periodic characteristics
in the residual part.
The following table shows that **integrating with the both part leads to best results.** 
(If the plot is not automatically visible, please see the file of `./plot/ablation_study.png`.)

<p align="center">
<img src=plot/ablation_study.png width="1000" height="180"/>
</p>


### Efficiency Analysis
We measure the efficiency of the TSINR method, and show the results in the following table. The results indicate that
**the TSINR is pretty efficient and lightweight**. (If the plot is not automatically visible, please see the file of `./plot/efficiency.png`.)

<p align="center">
<img src=plot/efficiency.png width="700" height="120"/>
</p>


### Algorithm for TSINR Anomaly Detection Algorithm
We present the algorithm for TSINR Anomaly Detection Algorithm as below. 
(If the plot is not automatically visible, please see the file of `./plot/algorithm.png`.)

<p align="center">
<img src=plot/algorithm.png width="1000" height="350"/>
</p>



