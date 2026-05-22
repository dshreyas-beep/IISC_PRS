import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

class RolloutDataset(Dataset):
    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)
        self.obs = d["obs"].astype(np.float32)
        self.act = d["actions"].astype(np.float32)

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        return self.obs[idx], self.act[idx]

class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="rollouts_multi_v1.npz")
    ap.add_argument("--out", type=str, default="ml/policy_bc.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = RolloutDataset(args.data)

    obs_dim = ds.obs.shape[1]
    n_total = len(ds)
    n_val = int(0.1 * n_total)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    model = MLPPolicy(obs_dim=obs_dim, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # actions are (delta_heading, speed_scale)
    # We use MSE for both
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for obs, act in train_loader:
            obs = obs.to(device)
            act = act.to(device)
            pred = model(obs)
            loss = loss_fn(pred, act)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr_loss += loss.item() * obs.size(0)
        tr_loss /= (len(train_loader) * args.batch)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs, act in val_loader:
                obs = obs.to(device)
                act = act.to(device)
                pred = model(obs)
                val_loss += loss_fn(pred, act).item() * obs.size(0)
        val_loss /= max(1, len(val_ds))

        print(f"Epoch {epoch:02d} | train MSE {tr_loss:.6f} | val MSE {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"model_state": model.state_dict(), "obs_dim": obs_dim, "hidden": args.hidden},
                args.out
            )
            print(f"  ✅ saved best model to {args.out}")

if __name__ == "__main__":
    main()
