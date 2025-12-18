DATA_PATHS_Attention_LSTM = {
    # "train_samples": "/media/stu231428/1120 7818/Master_github/datasets/training_set.pkl",
    # "val_samples":   "/media/stu231428/1120 7818/Master_github/datasets/validation_set.pkl",
    "model_out":     "/trained_models/attention_lstm.pt",
    "losses_out":    "/trained_models/train_plots/attention_lstm_losses.pkl",
    "plot_out":      "trained_models/train_plots/attention_lstm_training_plot.png",
}

HPARAMS_Attention_LSTM = {
    "input_size": 14,
    "hidden_dim": 256,
    
    "dropout": 0.4,
    "batch_size": 32,
    "lr": 0.00040088682195429254, 
    "weight_decay": 1.0214569609572818e-05,
    "num_epochs": 100,
    "patience": 10,
}