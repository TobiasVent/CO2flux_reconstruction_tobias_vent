# CO2flux_reconstruction_tobias_vent

Note that the current implementation assumes **Experiment 1**. If a different
experiment is used, the corresponding settings must be updated in the script.

To compute the training and validation samples, execute the script  
`data_preprocessing_training_validation_sets.py`.

Within the script, the fraction used for spatial subsampling as well as the
temporal range defining the training and validation sets can be specified
directly in the function calls.
In addition, the file paths for the training and validation datasets must be
specified in `config.data_paths.py`.

To generate test data, open `data_preprocessing_test_set.py` and specify the
desired time span and region. The script produces one sample file for each
selected year.

Hyperparameter optimization is performed using `optuna_pipeline.py`. The optimal
hyperparameters are then stored in the respective model configuration files.
Model training is carried out by executing `train_lstm.py`,
`train_mlp.py`, `train_xgboost.py`, and `train_attention_lstm.py`.

## CO₂ Flux Reconstruction from Test Data

The reconstruction of CO₂ flux fields is performed using the generated test datasets.
For this purpose, the reconstruction scripts

- `reconstruct_test_set_cache_lstm.py`
- `reconstruct_test_set_cache_xgboost.py`
- (and analogous scripts for other models)

are executed.

These scripts apply the trained models to the test data and create **yearly cache files**
containing spatially and temporally resolved CO₂ flux information.

### Reconstruction Procedure

- The **path to the test dataset** must be specified in the script.
- A **start year** and **end year** define the temporal range for which caches are generated.
- For each year in this range, model predictions are computed and stored separately.

### Cache Contents

Each yearly cache includes the following variables:

- latitude  
- longitude  
- timestep  
- simulated CO₂ flux  
- reconstructed CO₂ flux  

The generated caches are used for further analysis, visualization, and comparison
between simulated and reconstructed CO₂ fluxes.
