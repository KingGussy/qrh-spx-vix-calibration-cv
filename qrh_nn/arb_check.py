from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

#from src.qrh_sim.pricing_utils import norm_cdf

from qrh_nn.eval_utils import _device_or_default, load_model_and_norm, predict_raw, _jsonable, K_SPX_FIXED


REPO = Path(__file__).resolve().parents[1]

PACKED_PATH = REPO / "data" / "synthetic_qrh_spx_vix" / "full_data.npz"
TEST_IDX_PATH = REPO / "models" / "split" / "test_idx.npy"
OUT_DIR = REPO / "outputs" / "static_arb_compare"

MODELS = [
    {
        "name": "baseline",
        "label": "Baseline",
        "color": "tomato",
        "model_type": "fixed",
        "ckpt": None,
    },
    {
        "name": "arb_lam01",
        "label": "Arb λ=0.1",
        "color": "mediumorchid",
        "model_type": "fixed",
        "ckpt": REPO / "models" / "full_mtp" / "checkpoints_resmlp_arb_spxconv_lam01" / "best.pt",
    },
    {
        "name": "arb_lam10",
        "label": "Arb λ=1.0",
        "color": "deepskyblue",
        "model_type": "fixed",
        "ckpt": REPO / "models" / "full_mtp" / "checkpoints_resmlp_arb_spxconv_lam10" / "best.pt",
    },
    {
        "name": "arb_lam50",
        "label": "Arb λ=5.0",
        "color": "limegreen",
        "model_type": "fixed",
        "ckpt": REPO / "models" / "full_mtp" / "checkpoints_resmlp_arb_spxconv_lam50" / "best.pt",
    },
]

# pricing stuff on arrays
def norm_cdf(x: np.ndarray) -> np.ndarray:
    from math import erf, sqrt
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))

def black_call_from_forward(F: np.ndarray, K: np.ndarray, T: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    eps = 1e-12
    sigma = np.maximum(sigma, eps)
    sqrtT = np.sqrt(np.maximum(T, eps))
    volT = sigma * sqrtT
    d1 = (np.log(np.maximum(F, eps) / np.maximum(K, eps)) + 0.5 * volT * volT) / np.maximum(volT, eps)
    d2 = d1 - volT
    return F * norm_cdf(d1) - K * norm_cdf(d2)


def fit_error_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    err = y_pred - y_true
    abs_err = np.abs(err)
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(abs_err)),
        "q95_abs": float(np.quantile(abs_err.reshape(-1), 0.95)),
        "max_abs": float(np.max(abs_err)),
        "n_points": int(err.size),
    }


def smile_static_arb_diagnostics(
    k_grid: np.ndarray,
    iv: np.ndarray,
    T: float,
    F: float = 1.0,
) -> Dict[str, float]:
    k_grid = np.asarray(k_grid, dtype=np.float64)
    iv = np.asarray(iv, dtype=np.float64)

    K = F * np.exp(k_grid)
    C = black_call_from_forward(F=np.full_like(K, F, dtype=np.float64), K=K, T=np.full_like(K, T, dtype=np.float64), sigma=iv)

    mono_raw = C[1:] - C[:-1]
    mono_viols = np.maximum(mono_raw, 0.0)

    dK = K[1:] - K[:-1]
    slopes = (C[1:] - C[:-1]) / dK
    conv_raw = slopes[:-1] - slopes[1:]
    conv_viols = np.maximum(conv_raw, 0.0)

    return {
        "n_points": int(len(iv)),
        "mono_count": int(np.sum(mono_viols > 0.0)),
        "mono_max": float(mono_viols.max()) if mono_viols.size else 0.0,
        "mono_mean_pos": float(mono_viols[mono_viols > 0.0].mean()) if np.any(mono_viols > 0.0) else 0.0,
        "conv_count": int(np.sum(conv_viols > 0.0)),
        "conv_max": float(conv_viols.max()) if conv_viols.size else 0.0,
        "conv_mean_pos": float(conv_viols[conv_viols > 0.0].mean()) if np.any(conv_viols > 0.0) else 0.0,
    }


def aggregate_diag(diags: List[Dict[str, float]]) -> Dict[str, float]:
    mono_counts = np.array([d["mono_count"] for d in diags], dtype=np.float64)
    conv_counts = np.array([d["conv_count"] for d in diags], dtype=np.float64)
    mono_max = np.array([d["mono_max"] for d in diags], dtype=np.float64)
    conv_max = np.array([d["conv_max"] for d in diags], dtype=np.float64)

    return {
        "n_smiles": int(len(diags)),
        "frac_any_mono": float(np.mean(mono_counts > 0)),
        "frac_any_conv": float(np.mean(conv_counts > 0)),
        "mean_mono_count": float(np.mean(mono_counts)),
        "mean_conv_count": float(np.mean(conv_counts)),
        "p95_mono_max": float(np.quantile(mono_max, 0.95)),
        "p95_conv_max": float(np.quantile(conv_max, 0.95)),
        "max_mono_max": float(np.max(mono_max)),
        "max_conv_max": float(np.max(conv_max)),
    }


def per_smile_conv_stats(
    k_grid: np.ndarray,
    iv_batch: np.ndarray,
    T_list: np.ndarray,
) -> Tuple[List[Dict[str, float]], np.ndarray]:
    diags = []
    conv_max = np.zeros(len(iv_batch), dtype=np.float64)
    for i in range(len(iv_batch)):
        d = smile_static_arb_diagnostics(
            k_grid=k_grid,
            iv=iv_batch[i],
            T=float(T_list[i]),
            F=1.0,
        )
        diags.append(d)
        conv_max[i] = d["conv_max"]
    return diags, conv_max


def evaluate_one_model(
    *,
    model_name: str,
    model_label: str,
    color: str,
    model_type: str,
    ckpt_override: Path | None,
    device: torch.device,
    X_test: np.ndarray,
    Y_test: np.ndarray,
) -> Dict[str, object]:
    model, cfg, X_mu, X_sd, Y_mu, Y_sd = load_model_and_norm(
        model_type,
        device=device,
        ckpt_override=ckpt_override,
    )

    Y_pred = predict_raw(model, X_test, X_mu, X_sd, Y_mu, Y_sd, device=device)

    fit_all = fit_error_metrics(Y_pred, Y_test)
    fit_spx = fit_error_metrics(Y_pred[:, :15], Y_test[:, :15])
    fit_vix = fit_error_metrics(Y_pred[:, 15:], Y_test[:, 15:])

    iv_pred_spx = Y_pred[:, :15]
    iv_true_spx = Y_test[:, :15]
    T_list = X_test[:, 15].astype(np.float64)

    pred_diags, pred_conv_max = per_smile_conv_stats(K_SPX_FIXED, iv_pred_spx, T_list)
    true_diags, _ = per_smile_conv_stats(K_SPX_FIXED, iv_true_spx, T_list)

    return {
        "model_name": model_name,
        "model_label": model_label,
        "color": color,
        "fit_all": fit_all,
        "fit_spx": fit_spx,
        "fit_vix": fit_vix,
        "predicted_summary": aggregate_diag(pred_diags),
        "target_summary": aggregate_diag(true_diags),
        "_pred_conv_max": pred_conv_max,
        "_Y_pred": Y_pred,
    }


def save_combined_table(results: List[Dict[str, object]], out_dir: Path) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "model_name": r["model_name"],
            "model_label": r["model_label"],
            "rmse_all": r["fit_all"]["rmse"],
            "mae_all": r["fit_all"]["mae"],
            "q95_all": r["fit_all"]["q95_abs"],
            "rmse_spx": r["fit_spx"]["rmse"],
            "mae_spx": r["fit_spx"]["mae"],
            "q95_spx": r["fit_spx"]["q95_abs"],
            "rmse_vix": r["fit_vix"]["rmse"],
            "mae_vix": r["fit_vix"]["mae"],
            "q95_vix": r["fit_vix"]["q95_abs"],
            "frac_any_conv": r["predicted_summary"]["frac_any_conv"],
            "mean_conv_count": r["predicted_summary"]["mean_conv_count"],
            "p95_conv_max": r["predicted_summary"]["p95_conv_max"],
            "max_conv_max": r["predicted_summary"]["max_conv_max"],
            "frac_any_mono": r["predicted_summary"]["frac_any_mono"],
            "mean_mono_count": r["predicted_summary"]["mean_mono_count"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "comparison_table.csv", index=False)

    combined_payload = {
        r["model_name"]: {
            "model_label": r["model_label"],
            "fit_all": r["fit_all"],
            "fit_spx": r["fit_spx"],
            "fit_vix": r["fit_vix"],
            "predicted_summary": r["predicted_summary"],
            "target_summary": r["target_summary"],
        }
        for r in results
    }
    with open(out_dir / "combined_summary.json", "w") as f:
        json.dump(_jsonable(combined_payload), f, indent=2)

    return df


def plot_tradeoff_scatter(df: pd.DataFrame, results: List[Dict[str, object]], out_path: Path) -> None:
    color_map = {r["model_label"]: r["color"] for r in results}
    plt.figure(figsize=(6.5, 5.5))
    for _, row in df.iterrows():
        plt.scatter(
            row["rmse_spx"],
            row["frac_any_conv"],
            s=80,
            color=color_map[row["model_label"]],
        )
        plt.text(
            row["rmse_spx"],
            row["frac_any_conv"],
            f"  {row['model_label']}",
            va="center",
        )
    plt.xlabel("SPX RMSE (IV)")
    plt.ylabel("Fraction of smiles with convexity violation")
    plt.title("Fit vs convexity tradeoff")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()


def pick_example_indices(
    baseline_pred_conv_max: np.ndarray,
    n_examples: int = 3,
) -> List[int]:
    viol_idx = np.where(baseline_pred_conv_max > 0)[0]
    if len(viol_idx) == 0:
        return [0]

    vals = baseline_pred_conv_max[viol_idx]
    order = np.argsort(vals)

    picks = []
    picks.append(int(viol_idx[order[len(order) // 2]]))
    picks.append(int(viol_idx[order[min(len(order) - 1, int(0.95 * len(order)))]]))
    picks.append(int(viol_idx[order[-1]]))

    out = []
    seen = set()
    for p in picks:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out[:n_examples]


def plot_example_smiles(
    *,
    example_indices: List[int],
    results: List[Dict[str, object]],
    Y_test: np.ndarray,
    X_test: np.ndarray,
    out_dir: Path,
) -> None:
    label_map = {r["model_name"]: r["model_label"] for r in results}
    color_map = {r["model_name"]: r["color"] for r in results}
    pred_map = {r["model_name"]: r["_Y_pred"] for r in results}

    for idx in example_indices:
        plt.figure(figsize=(8.5, 4.8))
        plt.plot(
            K_SPX_FIXED,
            Y_test[idx, :15],
            linewidth=1.5,
            label="Target",
            color="black",
        )

        for r in results:
            y_pred = pred_map[r["model_name"]][idx, :15]
            plt.plot(
                K_SPX_FIXED,
                y_pred,
                linewidth=1.5,
                label=label_map[r["model_name"]],
                color=color_map[r["model_name"]],
            )

        T = float(X_test[idx, 15])
        plt.xlabel("SPX log-moneyness k")
        plt.ylabel("Implied vol")
        plt.title(f"SPX smile overlay | test idx={idx} | T={T:.4f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"smile_overlay_idx_{idx}.png", dpi=150)
        plt.show()


def main():
    device = _device_or_default(None)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with np.load(PACKED_PATH, allow_pickle=False) as z:
        X = z["X"].astype(np.float32, copy=False)
        Y = z["Y"].astype(np.float32, copy=False)

    test_idx = np.load(TEST_IDX_PATH).astype(np.int64)
    X_test = X[test_idx]
    Y_test = Y[test_idx]

    results: List[Dict[str, object]] = []
    for spec in MODELS:
        print(f"Running model: {spec['label']}")
        results.append(
            evaluate_one_model(
                model_name=spec["name"],
                model_label=spec["label"],
                color=spec["color"],
                model_type=spec["model_type"],
                ckpt_override=spec["ckpt"],
                device=device,
                X_test=X_test,
                Y_test=Y_test,
            )
        )

    df = save_combined_table(results, OUT_DIR)

    print("\nComparison table:")
    print(df.to_string(index=False))

    plot_tradeoff_scatter(df, results, OUT_DIR / "scatter_fit_vs_conv.png")

    baseline = next(r for r in results if r["model_name"] == "baseline")
    example_indices = pick_example_indices(baseline["_pred_conv_max"], n_examples=3)
    plot_example_smiles(
        example_indices=example_indices,
        results=results,
        Y_test=Y_test,
        X_test=X_test,
        out_dir=OUT_DIR,
    )

    print("\nSaved outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()