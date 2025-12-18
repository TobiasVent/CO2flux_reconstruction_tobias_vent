import os
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple
from configs.data_paths import Stats_Data_Path
from configs.attention_lstm_config import DATA_PATHS_Attention_LSTM, HPARAMS_Attention_LSTM
from models.attention_lstm import LSTMModelAttentionTemporal

Sample = namedtuple("Sample", ["features", "target", "meta"])


#check the years to reconstruct
START_YEAR = 2012
END_YEAR = 2018

region = "global"
experiment = "experiment_1"
#choose the test set path pattern
TEST_PATH_PATTERN = (
    "data/test_sest/{experiment}/{region}"
    "{region}_test_{year}_{experiment}.pkl"
)

#choose output cache directory and pattern
CACHE_DIR = f"/data/reconstruction_cache/{experiment}/{region}"
OUT_PATTERN = os.path.join(CACHE_DIR, "Attention_LSTM_{region}_reconstruction_{year}_{experiment}.pkl")


def collate_with_meta(batch):
    xs, ys, metas = zip(*batch)
    xs = torch.stack(xs, dim=0)
    ys = torch.stack(ys, dim=0).view(-1)
    return xs, ys, list(metas)


class PointwiseSampleDatasetWithMeta(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        X = torch.tensor(s.features, dtype=torch.float32)
        y = torch.tensor(s.target, dtype=torch.float32)
        return X, y, s.meta


def reconstruct_one_year(model, device, year, target_mean, target_std):
    test_path = TEST_PATH_PATTERN.format(year=year,region = region,experiment = experiment)
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test set not found for year {year}: {test_path}")

    with open(test_path, "rb") as f:
        samples = pickle.load(f)

    ds = PointwiseSampleDatasetWithMeta(samples)
    loader = DataLoader(
        ds,
        batch_size=2000,
        shuffle=False,
        collate_fn=collate_with_meta
    )

    rows = []
    with torch.no_grad():
        for xb, yb, metas in loader:
            xb = xb.to(device)
            preds = model(xb).cpu().view(-1)
            yb = yb.cpu().view(-1)

            # denormalize
            preds = preds * target_std + target_mean
            yb = yb * target_std + target_mean

            for i in range(len(preds)):
                rows.append({
                    "nav_lat": metas[i]["nav_lat"],
                    "nav_lon": metas[i]["nav_lon"],
                    "time_counter": metas[i]["time_counter"],
                    "co2flux_pre": float(yb[i]),
                    "reconstructed_co2flux_pre": float(preds[i]),
                })

    df = (
        pd.DataFrame(rows)
        .sort_values(["nav_lat", "nav_lon", "time_counter"])
        .reset_index(drop=True)
    )

    out_path = OUT_PATTERN.format(year=year,region = region, experiment = experiment)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_pickle(out_path)

    print(f"Saved {year}: {out_path} | shape={df.shape}")


def main():
    if END_YEAR < START_YEAR:
        raise ValueError(f"Invalid range: START_YEAR={START_YEAR} > END_YEAR={END_YEAR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Range:", START_YEAR, "-", END_YEAR)

    os.makedirs(CACHE_DIR, exist_ok=True)

    # load normalization stats
    #this should be the same as used during training


    training_stats_dir = pickle.load(open("data/training_set/training_stats.pkl", "rb"))

    target_mean = training_stats_dir["target_mean"]
    target_std = training_stats_dir["target_stds"]
    # target_mean = 0.1847209
    # target_std = 1.3368018

    # load model once
    model = LSTMModelAttentionTemporal(
        input_size=HPARAMS_Attention_LSTM["input_size"],
        hidden_dim=HPARAMS_Attention_LSTM["hidden_dim"],
        dropout=HPARAMS_Attention_LSTM["dropout"],
    ).to(device)

    model.load_state_dict(torch.load(DATA_PATHS_Attention_LSTM["model_out"], map_location=device))
    model.eval()

    # run yearly
    for year in range(START_YEAR, END_YEAR + 1):
        reconstruct_one_year(model, device, year, target_mean, target_std)


if __name__ == "__main__":
    main()
