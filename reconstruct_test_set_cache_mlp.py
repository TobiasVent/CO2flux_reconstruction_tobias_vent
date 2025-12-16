import os
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple
import numpy as np

from configs.mlp_config import DATA_PATHS, HPARAMS
from models.mlp import MLPModel

Sample = namedtuple("Sample", ["features", "target", "meta"])


CACHE_DIR = "/media/stu231428/1120 7818/Master_github/datasets/cache"

TEST_PATH_PATTERN = (
    "/media/stu231428/1120 7818/Master_github/datasets/yearly/"
    "global_test_{year}_experiment_1.pkl"
)

# Zeitraum festlegen
START_YEAR = 2018
END_YEAR = 2018


def collate_with_meta(batch):
    xs, ys, metas = zip(*batch)
    xs = torch.stack(xs, dim=0)
    ys = torch.stack(ys, dim=0).view(-1)
    return xs, ys, list(metas)


def flatten_dyn_plus_static(x_window, n_dyn=10):
    x_dyn = x_window[:, :n_dyn].reshape(-1)
    x_static = x_window[0, n_dyn:]
    vec = np.concatenate([x_dyn, x_static]).astype(np.float32)
    return torch.tensor(vec, dtype=torch.float32)


class PointwiseMLPDatasetWithMeta(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x = flatten_dyn_plus_static(s.features, n_dyn=10)
        y = torch.tensor(s.target, dtype=torch.float32)
        return x, y, s.meta


def reconstruct_one_year(model, device, target_mean, target_std, year):
    test_path = TEST_PATH_PATTERN.format(year=year)
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test set not found for year {year}: {test_path}")

    out_path = os.path.join(CACHE_DIR, f"MLP_global_reconstruction_{year}_experiment_1.pkl")

    with open(test_path, "rb") as f:
        test_samples = pickle.load(f)

    dataset = PointwiseMLPDatasetWithMeta(test_samples)
    loader = DataLoader(
        dataset,
        batch_size=HPARAMS["batch_size"],
        shuffle=False,
        collate_fn=collate_with_meta,
    )

    rows = []
    n_batches = len(loader)
    n_total = len(dataset)
    seen = 0

    with torch.no_grad():
        for b_idx, (xb, yb, metas) in enumerate(loader, start=1):
            xb = xb.to(device)
            preds = model(xb).detach().cpu().view(-1)
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

            seen += len(preds)
            if b_idx == 1 or b_idx % 1000 == 0 or b_idx == n_batches:
                print(f"Year {year}: batch {b_idx}/{n_batches}, {seen}/{n_total} samples processed")

    df = (
        pd.DataFrame(rows)
        .sort_values(["nav_lat", "nav_lon", "time_counter"])
        .reset_index(drop=True)
    )

    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_pickle(out_path)
    print(f"Saved year {year}: {out_path}, shape={df.shape}")


def main():
    if END_YEAR < START_YEAR:
        raise ValueError(f"Invalid range: START_YEAR={START_YEAR} > END_YEAR={END_YEAR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Range:", START_YEAR, "-", END_YEAR)

    os.makedirs(CACHE_DIR, exist_ok=True)

    # These should match training normalization
    target_mean = 0.1847209
    target_std = 1.3368018

    model = MLPModel(
        input_dim=HPARAMS["input_dim"],
        hidden_dims=HPARAMS["hidden_dims"],
        dropout=HPARAMS["dropout"],
    ).to(device)

    model.load_state_dict(torch.load(DATA_PATHS["model_out"], map_location=device))
    model.eval()

    for year in range(START_YEAR, END_YEAR + 1):
        reconstruct_one_year(model, device, target_mean, target_std, year)


if __name__ == "__main__":
    main()
