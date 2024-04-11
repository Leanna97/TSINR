# TSINR: Capturing Temporal Continuity via Implicit Neural Representations for Time Series Anomaly Detection

<p align="center">
<img src=./plot/motivation.png width="1100" height="300"/>
</p>

## Abstract
Time series anomaly detection aims to analyze time series data of a 
system to identify unusual patterns in the data or deviations from
the system’s expected behavior. The reconstruction-based methods
are the mainstream of the time series anomaly detection methods,
which exploit learning point-wise representation. However, the
unlabeled anomaly points in the training data may cause these
reconstruction-based methods to learn and reconstruct anomalous
data, resulting in the challenge of capturing normal patterns from
time series. To address this challenge, we propose a time series
anomaly detection method based on implicit neural representation
(INR) reconstruction, named TSINR, to alleviate the limitations of
the existing reconstruction-based methods. Due to the property
of spectral bias, TSINR enables prioritizing low-frequency signals
and exhibiting poorer performance on high-frequency abnormal
data. Specifically, we adopt INR to parameterize time series data
as continuous functions and employ a transformer-based architec-
ture to predict the INR of given time series data. As a result, the
proposed TSINR method achieves the advantage of capturing the
temporal continuity in time series data and thus is more sensitive
to discontinuous anomaly data. In addition, we further design a
novel continuous function to learn inter- and intra-variable infor-
mation, and leverage a pre-trained large language model to am-
plify the intense fluctuations in anomalies. Extensive experiments
demonstrate that the proposed TSINR method achieves superior
overall performance on both univariate and multivariate time series
anomaly detection benchmark compared to other state-of-the-art
reconstruction-based methods


## Reconstruction from INR
<p align="center">
<img src=./plot/architecture.png width="1100" height="500"/>
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
<img src=./plot/experiments_major.png width="1100" height="600"/>
</p>




