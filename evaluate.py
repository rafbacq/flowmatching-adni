"""
Unified evaluation summary.

Reads the artefacts written by the three model scripts and produces:
  - results/summary.json       -- single machine-readable metrics file
  - results/summary.png        -- 2x2 figure: CDE scatter (val), residuals,
                                  training curves, baseline-vs-CDE bar
  - results/model_comparison.png -- bar chart RMSE: mean-baseline vs FM-only
                                  vs YOLO-only vs FM+YOLO (linear probe)
                                  vs Neural CDE (the full model)

The linear probes give you a sanity floor: if the CDE doesn't beat a ridge
regression on concatenated FM+YOLO features, the time component is not pulling
its weight. These numbers go straight into the comparison chart.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = "adni_data_v3"
RESULTS_DIR = "results"


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def ridge_probe(X_train, y_train, X_val, y_val):
    mdl = Ridge(alpha=1.0).fit(X_train, y_train)
    p = mdl.predict(X_val)
    return {
        "mae": float(mean_absolute_error(y_val, p)),
        "rmse": rmse(y_val, p),
        "r2": float(r2_score(y_val, p)),
        "preds": p,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fm = np.load(f"{DATA_DIR}/fm_embeddings.npy")
    yo = np.load(f"{DATA_DIR}/yolo_embeddings.npy")
    targets = np.load(f"{DATA_DIR}/targets.npy")
    rids = np.load(f"{DATA_DIR}/rids.npy")
    with open(f"{DATA_DIR}/splits.json") as f:
        splits = json.load(f)
    with open(f"{DATA_DIR}/target_stats.json") as f:
        stats = json.load(f)
    train_rids = set(int(x) for x in splits.get("train_rids", []))
    mu = stats["adas13_mean"]

    train_mask = np.array([int(r) in train_rids for r in rids])
    val_mask = ~train_mask

    # ---- Baselines ----
    baselines = {}
    baselines["mean"] = {
        "mae": float(np.mean(np.abs(targets[val_mask] - mu))),
        "rmse": rmse(targets[val_mask], np.full(val_mask.sum(), mu)),
        "r2": float(r2_score(targets[val_mask], np.full(val_mask.sum(), mu))),
    }
    baselines["fm_ridge"] = ridge_probe(
        fm[train_mask], targets[train_mask], fm[val_mask], targets[val_mask]
    )
    baselines["yolo_ridge"] = ridge_probe(
        yo[train_mask], targets[train_mask], yo[val_mask], targets[val_mask]
    )
    feats = np.concatenate([fm, yo], axis=1)
    baselines["fm_yolo_ridge"] = ridge_probe(
        feats[train_mask], targets[train_mask], feats[val_mask], targets[val_mask]
    )

    # ---- Load CDE predictions ----
    cde_path = f"{RESULTS_DIR}/cde_predictions.npz"
    cde_metrics = {}
    if os.path.exists(cde_path):
        d = np.load(cde_path, allow_pickle=True)
        pred = d["pred"]; true = d["true"]; mask = d["mask"]; is_train = d["is_train"]
        val_pred = pred[~is_train][mask[~is_train]]
        val_true = true[~is_train][mask[~is_train]]
        cde_metrics = {
            "mae": float(mean_absolute_error(val_true, val_pred)),
            "rmse": rmse(val_true, val_pred),
            "r2": float(r2_score(val_true, val_pred)),
            "n": int(len(val_true)),
        }
    else:
        print(f"Warning: {cde_path} not found; run cde_advanced.py first.")

    summary = {
        "baselines": {k: {kk: v for kk, v in x.items() if kk != "preds"}
                      for k, x in baselines.items()},
        "neural_cde": cde_metrics,
        "n_train_pairs": int(train_mask.sum()),
        "n_val_pairs": int(val_mask.sum()),
    }
    with open(f"{RESULTS_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # ---- Model comparison bar chart ----
    labels, rmses = [], []
    for name in ["mean", "fm_ridge", "yolo_ridge", "fm_yolo_ridge"]:
        labels.append(name); rmses.append(baselines[name]["rmse"])
    if cde_metrics:
        labels.append("neural_cde"); rmses.append(cde_metrics["rmse"])

    plt.figure(figsize=(8, 4.5))
    colors = ["#888", "#4C78A8", "#F58518", "#54A24B", "#E45756"][: len(labels)]
    plt.bar(labels, rmses, color=colors, edgecolor="k")
    for i, v in enumerate(rmses):
        plt.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.ylabel("Validation RMSE (ADAS13 points)")
    plt.title("Model comparison on held-out patients")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/model_comparison.png", dpi=150)
    plt.close()

    # ---- Scatter: each baseline vs CDE on val ----
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # left: best ridge probe
    best = min(("fm_ridge", "yolo_ridge", "fm_yolo_ridge"), key=lambda n: baselines[n]["rmse"])
    yv = targets[val_mask]; pv = baselines[best]["preds"]
    lo, hi = min(yv.min(), pv.min()) - 1, max(yv.max(), pv.max()) + 1
    axs[0].scatter(yv, pv, alpha=0.5, s=16)
    axs[0].plot([lo, hi], [lo, hi], "k--", lw=1)
    axs[0].set_xlim(lo, hi); axs[0].set_ylim(lo, hi)
    axs[0].set_title(f"Ridge probe ({best})  RMSE={baselines[best]['rmse']:.2f}")
    axs[0].set_xlabel("true ADAS13"); axs[0].set_ylabel("predicted")
    axs[0].grid(True)

    if cde_metrics:
        axs[1].scatter(val_true, val_pred, alpha=0.5, s=16, color="C3")
        lo2 = min(val_true.min(), val_pred.min()) - 1
        hi2 = max(val_true.max(), val_pred.max()) + 1
        axs[1].plot([lo2, hi2], [lo2, hi2], "k--", lw=1)
        axs[1].set_xlim(lo2, hi2); axs[1].set_ylim(lo2, hi2)
        axs[1].set_title(f"Neural CDE  RMSE={cde_metrics['rmse']:.2f}  R2={cde_metrics['r2']:.2f}")
        axs[1].set_xlabel("true ADAS13"); axs[1].set_ylabel("predicted")
        axs[1].grid(True)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/summary.png", dpi=150)
    plt.close()
    print(f"\nFigures: {RESULTS_DIR}/model_comparison.png, {RESULTS_DIR}/summary.png")


if __name__ == "__main__":
    main()
