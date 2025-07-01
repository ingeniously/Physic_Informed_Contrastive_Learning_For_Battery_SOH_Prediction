# PICL
Physic Informed Contrastive Learning for Battery State of Health Prediction


# 1. System requirements
python version: 3.7.10
conda create -n picl python=3.7.10

|    Package     | Version  |
|:--------------:|:--------:|
|     torch      |  1.7.1   |
|    sklearn     |  0.24.2  |
|     numpy      |  1.20.3  |
|     pandas     |  1.3.5   |
|   matplotlib   |  3.3.4   |
|  scienceplots  |          |

# 2. How to run
Code running on the Nasa lithium-ion battery dataset [Li-ion Battery Aging Datasets](https://data.nasa.gov/dataset/li-ion-battery-aging-datasets)

1. Preprocess the data using `data_processing.py` in `data_analysis folderrrrrrrr`
2. Run the `dataloader.py` t load the data in tensors 
3. Run the `main.py` file to train our model. The program will generate a folder `results` and save the results in it.Look at the parser arguments in the main file to know the required argument to run the code ex: <pre>
```python /home/cedric/PICL/main.py --csv_file /home/choi/PICL/data_processed/preprocessed_battery_health_dataset_all_points.csv```
</pre>


 