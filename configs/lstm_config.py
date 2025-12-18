DATA_PATHS_LSTM = {
    "train_samples": "/media/stu231428/1120 7818/Master_github/datasets/training_set.pkl",
    "val_samples":   "/media/stu231428/1120 7818/Master_github/datasets/validation_set.pkl",
    "model_out":     "/data/stu231428/Master_Thesis/main/trained_models/lsmt_with_pos.pt",
    "losses_out":    "/media/stu231428/1120 7818/Master_github/datasets/training_plots/lstm_losses.pkl",
    "plot_out":      "/media/stu231428/1120 7818/Master_github/datasets/training_plots/lstm_training_plot.png",
}

HPARAMS_LSTM = {
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