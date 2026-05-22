# ml/train_bc_norm.py
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from simcore.rollout.rollout import ACTION_DELTA_MAX


class NpzDataset(Dataset):
    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)
        self.obs = d["obs"].astype(np.float32)
        self.act = d["actions"].astype(np.float32)

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        return self.obs[idx], self.act[idx]


class BoundedPolicy(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, 2)

    def forward(self, x):
        z = self.backbone(x)
        y = self.head(z)
        # y0 in [-1,1] via tanh -> scaled to [-ACTION_DELTA_MAX, ACTION_DELTA_MAX]
        delta = torch.tanh(y[:, 0:1]) * ACTION_DELTA_MAX
        # y1 in [0,1] via sigmoid
        speed = torch.sigmoid(y[:, 1:2])
        return torch.cat([delta, speed], dim=1)


def compute_norm(obs: np.ndarray):
    mean = obs.mean(axis=0)
    std = obs.std(axis=0)
    std = np.maximum(std, 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, default="ml/policy_bc_norm.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = NpzDataset(args.data)
    obs_dim = ds.obs.shape[1]

    # normalization stats from full dataset
    mean, std = compute_norm(ds.obs)

    n_total = len(ds)
    n_val = int(0.1 * n_total)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    def collate(batch):
        obs = np.stack([b[0] for b in batch]).astype(np.float32)
        act = np.stack([b[1] for b in batch]).astype(np.float32)
        obs = (obs - mean) / std
        return torch.from_numpy(obs), torch.from_numpy(act)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate)

    model = BoundedPolicy(obs_dim=obs_dim, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr = 0.0
        ntr = 0
        for obs, act in train_loader:
            obs = obs.to(device)
            act = act.to(device)

            pred = model(obs)
            loss = loss_fn(pred, act)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr += loss.item() * obs.size(0)
            ntr += obs.size(0)
        tr /= max(1, ntr)

        model.eval()
        va = 0.0
        nva = 0
        with torch.no_grad():
            for obs, act in val_loader:
                obs = obs.to(device)
                act = act.to(device)
                pred = model(obs)
                va += loss_fn(pred, act).item() * obs.size(0)
                nva += obs.size(0)
        va /= max(1, nva)

        print(f"Epoch {epoch:02d} | train MSE {tr:.6f} | val MSE {va:.6f}")

        if va < best:
            best = va
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "obs_dim": obs_dim,
                    "hidden": args.hidden,
                    "obs_mean": mean,
                    "obs_std": std,
                },
                args.out
            )
            print(f"  ✅ saved best model to {args.out}")


if __name__ == "__main__":
    main()
