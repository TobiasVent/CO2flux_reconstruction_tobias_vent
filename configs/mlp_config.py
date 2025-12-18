DATA_PATHS_MLP = {
    "train_samples": "/media/stu231428/1120 7818/Master_github/datasets/training_set.pkl",
    "val_samples":   "/media/stu231428/1120 7818/Master_github/datasets/validation_set.pkl",
    "model_out":     "/media/stu231428/1120 7818/Master_github/datasets/trained_models/mlp.pt",
    "losses_out":    "/media/stu231428/1120 7818/Master_github/datasets/training_plots/mlp_losses.pkl",
    "plot_out":      "/media/stu231428/1120 7818/Master_github/datasets/training_plots/mlp_training_plot.png",
}

HPARAMS_MLP = {
    "input_dim": 44,
    "hidden_dims": [207,248,198],
    
    "dropout": 0.0,
    "batch_size": 128,
    "lr": 0.00030206699545840877, 
    "weight_decay": 1.9936923594451204e-05,
    "num_epochs": 100,
    "patience": 10,
}