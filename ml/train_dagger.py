# ml/train_dagger.py
import argparse
import numpy as np
import os
import subprocess
import sys

from simcore.rollout.sim import SingleAgentSim
from simcore.rollout.dagger import collect_dagger_data
from simcore.behavior.ml_policy import LoadedPolicy


def save_npz(path, obs, actions):
    np.savez_compressed(
        path,
        obs=obs.astype(np.float32),
        actions=actions.astype(np.float32),
        rewards=np.zeros((obs.shape[0],), dtype=np.float32),
        dones=np.zeros((obs.shape[0],), dtype=np.float32),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_data", type=str, required=True)
    ap.add_argument("--work", type=str, default="ml/dagger_work")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--steps_per_roll", type=int, default=800)
    ap.add_argument("--episodes_per_combo", type=int, default=3)
    ap.add_argument("--out_ckpt", type=str, default="ml/policy_dagger.pt")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    os.makedirs(args.work, exist_ok=True)

    seed = np.load(args.seed_data, allow_pickle=True)
    obs_agg = seed["obs"].astype(np.float32)
    act_agg = seed["actions"].astype(np.float32)

    tmp_npz = os.path.join(args.work, "agg.npz")
    tmp_ckpt = os.path.join(args.work, "policy_tmp.pt")

    species = ["elephant", "leopard", "sloth_bear", "tiger", "human"]
    topologies = ["flat", "edge_farmland", "corridor_fence", "scarce_water", "rugged"]

    for r in range(1, args.rounds + 1):
        print(f"\n===== DAgger round {r}/{args.rounds} =====")

        # Train on aggregated dataset using module execution so simcore imports work
        save_npz(tmp_npz, obs_agg, act_agg)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ml.train_bc_norm",
                "--data",
                tmp_npz,
                "--out",
                tmp_ckpt,
                "--epochs",
                str(args.epochs),
                "--hidden",
                str(args.hidden),
            ],
            check=True,
        )

        policy = LoadedPolicy(tmp_ckpt, device="cpu")

        new_obs = []
        new_act = []

        ep_seed = 1000 + 77 * r
        for topo in topologies:
            for sp in species:
                for _ in range(args.episodes_per_combo):
                    sim = SingleAgentSim(seed=ep_seed, species=sp, topology=topo)
                    obs, acts, _ = collect_dagger_data(sim, policy, steps=args.steps_per_roll, seed=ep_seed + 1)
                    new_obs.append(obs)
                    new_act.append(acts)
                    ep_seed += 3

        new_obs = np.concatenate(new_obs, axis=0)
        new_act = np.concatenate(new_act, axis=0)

        print("Added samples:", new_obs.shape[0])

        obs_agg = np.concatenate([obs_agg, new_obs], axis=0)
        act_agg = np.concatenate([act_agg, new_act], axis=0)

        print("Total agg:", obs_agg.shape[0])

    # Final train and save
    save_npz(tmp_npz, obs_agg, act_agg)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ml.train_bc_norm",
            "--data",
            tmp_npz,
            "--out",
            args.out_ckpt,
            "--epochs",
            str(args.epochs),
            "--hidden",
            str(args.hidden),
        ],
        check=True,
    )

    print("\n✅ Final DAgger policy saved to:", args.out_ckpt)


if __name__ == "__main__":
    main()
