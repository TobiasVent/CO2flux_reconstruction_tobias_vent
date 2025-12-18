# xgboost_config.py

# ========== Datenpfade ====================
DATA_PATHS_XGBoost = {
    # "train_samples": "/media/stu231428/1120 7818/Master_github/datasets/training_set.pkl",
    # "val_samples":   "/media/stu231428/1120 7818/Master_github/datasets/validation_set.pkl",
    "model_out":     "trained_models/xg_boost_with_pos_model.pkl",
    "losses_out":    "/trained_models/train_plots/xgboost_losses.pkl",
    "plot_out":      "/trained_models/train_plots/xgboost_training_plot.png",
}

# ========== Hyperparameter für XGBoost ====================
HPARAMS_XGBoost = {
    "booster": "dart",  # Typ des Boosters (DART: Dropout)
    "n_estimators": 400,  # Anzahl der Bäume
    "max_depth": 14,  # Maximale Tiefe der Bäume
    "learning_rate": 0.04292980473318146,  # Lernrate
    "subsample": 0.5459662212041141,  # Anteil der Daten, die pro Baum verwendet werden
    "colsample_bytree": 0.5336910753872645,  # Anteil der Features, die pro Baum verwendet werden
    "gamma": 0.016137253371362736,  # Minimale Reduktion des Verlustes, um einen weiteren Split zu machen
    "reg_lambda": 0.00023498796241233066,  # L2-Regularisierung
    "reg_alpha": 0.010524060361037127,  # L1-Regularisierung
    "min_child_weight": 4,  # Minimale Gewichtung für ein Kind, um einen Split zu machen
}
