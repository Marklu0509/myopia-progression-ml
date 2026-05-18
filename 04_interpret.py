"""
04_interpret.py
───────────────
【這個檔案在做什麼】
對最佳模型 (XGBoost) 做可解釋性分析，產出三張履歷用圖表：

  figures/shap_summary.png        — SHAP beeswarm plot（最重要的特徵是哪些）
  figures/calibration_plot.png    — 預測 vs 實際散點圖（模型校正好不好）
  figures/rct_benchmark.png       — 真實世界 vs 文獻 RCT 的年化進展對照

【為什麼用 SHAP】
SHAP (SHapley Additive exPlanations) 是目前 ML 可解釋性的業界標準。
它能告訴我們「每個特徵對每個病人的預測貢獻多少」，
比傳統 feature importance 更細緻、更有臨床意義。
放進履歷可以展示「會用 ML 但也懂得解釋給醫師聽」的能力。

【文獻對照圖的意義】
把真實世界病人的年化進展分布，對照發表 RCT 中的官方報告值，
讓讀履歷的人直接看到「這不只是模型練習，而是有臨床意義的比較」。
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble  import RandomForestRegressor
from sklearn.impute    import SimpleImputer
from sklearn.pipeline  import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠️  shap 未安裝，跳過 SHAP plot")

BASE      = Path(__file__).parent
FEAT_FILE = BASE / "features.csv"
FIG_DIR   = BASE / "figures"
FIG_DIR.mkdir(exist_ok=True)

FEATURES_A = [
    "age_start", "sex", "laterality",
    "myopia_d", "axl_baseline", "axl_dev_from_norm",
    "mi_rx", "atropine_conc",
    "early_slope_6m", "axl_std_6m",
]
FEAT_LABELS = {
    "age_start"         : "Age at baseline (yrs)",
    "sex"               : "Sex (M=1, F=0)",
    "laterality"        : "Laterality (OD=0, OS=1)",
    "myopia_d"          : "Baseline myopia (D)",
    "axl_baseline"      : "Baseline AXL (mm)",
    "axl_dev_from_norm" : "AXL deviation from norm (mm)",
    "mi_rx"             : "MiSight Rx power (D)",
    "atropine_conc"     : "Atropine concentration",
    "early_slope_6m"    : "Early AXL slope 0–6m (mm/mo)",
    "axl_std_6m"        : "AXL variability 0–6m (SD)",
}
TARGET = "y_annual_mm_yr"

# 文獻 RCT 年化進展 (mm/yr) — 僅眼軸
RCT_DATA = {
    "SV control\n(Liu 2021)"      : 0.36,
    "MiSight 1-yr\n(Chamberlain)" : 0.13,
    "MiSight 3-yr\n(Chamberlain)" : 0.15,
    "0.01% Atropine\n(ATOM2)"     : 0.28,
    "0.1% Atropine\n(LAMP)"       : 0.19,
    "0.01% Atro+SCL\n(This study)": None,   # 從真實資料填入
}


def get_best_model():
    """訓練最佳模型（XGBoost，若無則用 Random Forest）。"""
    if HAS_XGB:
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("mdl", XGBRegressor(
                n_estimators=200, max_depth=3,
                learning_rate=0.05, subsample=0.8,
                random_state=42, verbosity=0,
            )),
        ]), "XGBoost"
    else:
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("mdl", RandomForestRegressor(
                n_estimators=200, max_depth=4, random_state=42
            )),
        ]), "Random Forest"


# ══════════════════════════════════════════════════════════════════
# 圖 1：SHAP Summary (beeswarm)
# ══════════════════════════════════════════════════════════════════
def plot_shap(model_pipe, X: np.ndarray, feat_names: list, path: Path, model_name: str):
    if not HAS_SHAP:
        return

    # 先跑 imputer，再取出底層模型
    imp = model_pipe.named_steps["imp"]
    X_imp = imp.transform(X)
    mdl = model_pipe.named_steps["mdl"]

    # 相容 xgboost 3.x：用 shap.TreeExplainer + booster 物件
    try:
        booster     = mdl.get_booster()
        explainer   = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(X_imp)
    except Exception:
        # fallback: 用 predict-based KernelExplainer (慢但通用)
        bg = shap.sample(X_imp, 50, random_state=42)
        explainer   = shap.KernelExplainer(mdl.predict, bg)
        shap_values = explainer.shap_values(X_imp, nsamples=100)

    display_names = [FEAT_LABELS.get(f, f) for f in feat_names]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    shap.summary_plot(
        shap_values, X_imp,
        feature_names=display_names,
        show=False, plot_type="dot",
        color_bar_label="Feature value",
        max_display=10,
    )
    plt.title(
        f"SHAP Feature Importance — {model_name}\n"
        "Myopia Axial Progression Prediction",
        fontsize=12, fontweight="bold", pad=10
    )
    plt.xlabel("SHAP value (impact on predicted mm/yr)", fontsize=10)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ SHAP plot：{path}")


# ══════════════════════════════════════════════════════════════════
# 圖 2：Calibration (預測 vs 實際)
# ══════════════════════════════════════════════════════════════════
def plot_calibration(y_true, y_pred, model_name: str, path: Path):
    from sklearn.metrics import mean_squared_error, r2_score
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5.5))

    ax.scatter(y_true, y_pred, alpha=0.6, s=40,
               color="#2E75B6", edgecolors="white", linewidths=0.4, zorder=3)

    lims = [min(y_true.min(), y_pred.min()) - 0.05,
            max(y_true.max(), y_pred.max()) + 0.05]
    ax.plot(lims, lims, "r--", linewidth=1.2, label="Perfect calibration", zorder=2)

    # 趨勢線
    m, b = np.polyfit(y_true, y_pred, 1)
    xs = np.linspace(*lims, 100)
    ax.plot(xs, m * xs + b, color="#1F4E79", linewidth=1.5,
            label=f"Fitted (slope={m:.2f})", zorder=2)

    ax.set_xlabel("Observed AXL progression (mm/yr)", fontsize=11)
    ax.set_ylabel("Predicted AXL progression (mm/yr)", fontsize=11)
    ax.set_title(
        f"Calibration Plot — {model_name}\n"
        f"RMSE = {rmse:.4f} mm/yr   R² = {r2:.4f}",
        fontsize=12, fontweight="bold", pad=10
    )
    ax.legend(fontsize=9)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(linestyle="--", alpha=0.35)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Calibration plot：{path}")


# ══════════════════════════════════════════════════════════════════
# 圖 3：Real-world vs RCT Benchmark
# ══════════════════════════════════════════════════════════════════
def plot_rct_benchmark(y_real: np.ndarray, path: Path):
    real_median = float(np.median(y_real))
    real_q25    = float(np.percentile(y_real, 25))
    real_q75    = float(np.percentile(y_real, 75))

    # 把「This study」填入真實中位數
    rct = dict(RCT_DATA)
    rct["0.01% Atro+SCL\n(This study)"] = real_median

    labels  = list(rct.keys())
    values  = [rct[k] for k in labels]

    colors = [
        "#BDBDBD",   # SV control — 灰（baseline）
        "#2E75B6",   # MiSight 1yr
        "#1F4E79",   # MiSight 3yr
        "#70AD47",   # ATOM2
        "#92D050",   # LAMP
        "#C00000",   # This study — 紅色突出
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white",
                  width=0.6, zorder=3)

    # This study 加 IQR 誤差棒
    this_idx = len(labels) - 1
    ax.errorbar(
        this_idx, real_median,
        yerr=[[real_median - real_q25], [real_q75 - real_median]],
        fmt="none", color="#7F0000", capsize=6, linewidth=2, zorder=4
    )

    # 數值標籤
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.005,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=10,
            fontweight="bold" if i == this_idx else "normal",
            color="#C00000" if i == this_idx else "#333333"
        )

    ax.set_ylabel("Annualized AXL Progression (mm/yr)", fontsize=11)
    ax.set_title(
        "Real-World vs Published RCT Benchmarks\n"
        "Axial Length Progression Under Myopia Management",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.set_ylim(0, max(values) * 1.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.tick_params(axis="x", labelsize=8.5)

    # 圖例
    patches = [
        mpatches.Patch(color="#BDBDBD", label="Single Vision (control)"),
        mpatches.Patch(color="#2E75B6", label="MiSight SCL"),
        mpatches.Patch(color="#70AD47", label="Atropine"),
        mpatches.Patch(color="#C00000", label="This study (median ± IQR)"),
    ]
    ax.legend(handles=patches, fontsize=8.5, loc="upper right")

    # 來源備注
    ax.text(0.01, 0.01,
            "Sources: Chamberlain et al. 2019 (MiSight); Yam et al. 2019 (LAMP); "
            "Chia et al. 2012 (ATOM2); Liu et al. 2021 (SV)",
            transform=ax.transAxes, fontsize=7, color="#888888",
            verticalalignment="bottom")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ RCT benchmark 圖：{path}")


# ══════════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════════
def main():
    df = pd.read_csv(FEAT_FILE)
    feats = [f for f in FEATURES_A if f in df.columns]
    sub   = df[feats + [TARGET]].dropna(subset=[TARGET])

    X = sub[feats].values
    y = sub[TARGET].values

    print(f"訓練集：n={len(y)} 隻眼")

    model_pipe, model_name = get_best_model()
    model_pipe.fit(X, y)
    y_pred = model_pipe.predict(X)   # in-sample (for calibration visualization)

    # 圖 1：SHAP
    plot_shap(model_pipe, X, feats, FIG_DIR / "shap_summary.png", model_name)

    # 圖 2：Calibration
    plot_calibration(y, y_pred, model_name, FIG_DIR / "calibration_plot.png")

    # 圖 3：RCT benchmark
    plot_rct_benchmark(y, FIG_DIR / "rct_benchmark.png")

    print("\n✓ Layer 3 完成。所有圖表在 figures/")


if __name__ == "__main__":
    main()
