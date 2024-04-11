# TSINR: Capturing Temporal Continuity via Implicit Neural Representations for Time Series Anomaly Detection

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




