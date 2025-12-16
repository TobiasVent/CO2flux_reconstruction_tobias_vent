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

