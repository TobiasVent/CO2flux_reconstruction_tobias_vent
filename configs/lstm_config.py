DATA_PATHS = {
    "train_samples": "/media/stu231428/1120 7818/Master_github/datasets/training_set.pkl",
    "val_samples":   "/media/stu231428/1120 7818/Master_github/datasets/validation_set.pkl",
    "model_out":     "/media/stu231428/1120 7818/Master_github/datasets/trained_models/lstm.pt",
    "losses_out":    "/media/stu231428/1120 7818/Master_github/datasets/training_plots/lstm_losses.pkl",
    "plot_out":      "/media/stu231428/1120 7818/Master_github/datasets/training_plots/lstm_training_plot.png",
}

HPARAMS = {
    "input_size": 14,
    "hidden_dim": 128,
    "num_layers": 3,
    "dropout": 0.0,
    "batch_size": 32,
    "lr": 0.00015007036384757517,
    "weight_decay": 2.939570672798765e-06,
    "num_epochs": 100,
    "patience": 10,
}