import numpy as np
import torch
import torch.nn as nn

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
    ckpt = torch.load("ml/policy_bc.pt", map_location="cpu")
    model = MLPPolicy(ckpt["obs_dim"], ckpt["hidden"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    d = np.load("rollouts_multi_v1.npz", allow_pickle=True)
    obs = d["obs"].astype(np.float32)

    x = torch.from_numpy(obs[:5])
    with torch.no_grad():
        pred = model(x).numpy()

    print("Pred actions (first 5):")
    print(pred)

if __name__ == "__main__":
    main()
