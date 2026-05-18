"""
05_visualize.py
───────────────
【這個檔案在做什麼】
產出 6 張進階視覺化圖表，從多個臨床角度分析資料：

  figures/age_boxplot.png          — 各年齡組進展分布（boxplot + 樣本數）
  figures/age_scatter.png          — 連續年齡 vs 年化進展（scatter + 回歸線）
  figures/atropine_effect.png      — Atropine 劑量效應分析
  figures/rct_benchmark_v2.png     — 升級版 RCT 對照（加年齡資訊）
  figures/shap_age_interaction.png — SHAP dependency：early slope × age
  figures/source_comparison.png    — 兩診所資料一致性確認

【為什麼這些圖對履歷有用】
- 年齡分析圖直接回答最重要的臨床問題：「幾歲的孩子進展最快？」
- 升級版 RCT 對照圖讓面試官看到你懂得「蘋果比蘋果」——
  年齡中位數不同的試驗不能直接比較，放出來代表你有 domain awareness
- SHAP dependency 展示特徵交互作用，是進階 ML 技巧
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from sklearn.impute  import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

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
TARGET = "y_annual_mm_yr"

# 統一配色
C_MAIN   = "#1F4E79"
C_ACCENT = "#C00000"
C_GREEN  = "#375623"
C_GRAY   = "#BDBDBD"
C_LIGHT  = "#BDD7EE"

AGE_BINS   = [0, 8, 10, 12, 14, 16, 100]
AGE_LABELS = ["≤8", "9–10", "11–12", "13–14", "15–16", "≥17"]
AGE_COLORS = ["#4472C4","#2E75B6","#C00000","#ED7D31","#70AD47","#7030A0"]


# ══════════════════════════════════════════════════════════
# 圖 1：年齡分組 Boxplot
# ══════════════════════════════════════════════════════════
def plot_age_boxplot(df: pd.DataFrame, path: Path):
    df = df.copy()
    df["age_group"] = pd.cut(df["age_start"], bins=AGE_BINS,
                             labels=AGE_LABELS, right=True)
    groups = [df[df["age_group"] == g]["y_annual_mm_yr"].dropna().values
              for g in AGE_LABELS]
    counts = [len(g) for g in groups]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bp = ax.boxplot(
        groups, patch_artist=True, notch=False,
        widths=0.5, showfliers=True,
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
        medianprops=dict(color="white", linewidth=2.5),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )
    for patch, color in zip(bp["boxes"], AGE_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)

    # 中位數數值 + 樣本數標籤
    for i, (grp_data, cnt) in enumerate(zip(groups, counts), start=1):
        if len(grp_data) == 0:
            continue
        med = np.median(grp_data)
        ax.text(i, med + 0.03, f"{med:.2f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc=AGE_COLORS[i-1], alpha=0.85, ec="none"))
        ax.text(i, ax.get_ylim()[0] if ax.get_ylim()[0] > -0.8 else -0.65,
                f"n={cnt}", ha="center", va="top", fontsize=8, color="#555555")

    # 文獻參考線
    ax.axhline(0.13, color="#2E75B6", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axhline(0.36, color=C_GRAY,    linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(6.45, 0.135, "MiSight RCT\n(0.13)", fontsize=7.5, color="#2E75B6",
            va="bottom", ha="right")
    ax.text(6.45, 0.365, "SV control\n(0.36)", fontsize=7.5, color=C_GRAY,
            va="bottom", ha="right")

    ax.set_xticks(range(1, len(AGE_LABELS)+1))
    ax.set_xticklabels(AGE_LABELS, fontsize=10)
    ax.set_xlabel("Age Group at Treatment Start (years)", fontsize=11)
    ax.set_ylabel("Annualized AXL Progression (mm/yr)", fontsize=11)
    ax.set_title(
        "Myopia Progression by Age Group\n"
        "MiSight ± Atropine — Real-World Cohort",
        fontsize=12, fontweight="bold", pad=10
    )
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ {path.name}")


# ══════════════════════════════════════════════════════════
# 圖 2：連續年齡 Scatter + 回歸線
# ══════════════════════════════════════════════════════════
def plot_age_scatter(df: pd.DataFrame, path: Path):
    sub = df[["age_start","y_annual_mm_yr","atropine_conc"]].dropna()
    age = sub["age_start"].values
    y   = sub["y_annual_mm_yr"].values
    atro = sub["atropine_conc"].values

    # 按 atropine 分組著色
    color_map = {0.0: C_LIGHT, 0.01: "#70AD47", 0.05: "#ED7D31", 0.125: C_ACCENT, 0.02: "#7030A0"}
    colors = [color_map.get(a, C_GRAY) for a in atro]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(age, y, c=colors, alpha=0.65, s=45,
               edgecolors="white", linewidths=0.4, zorder=3)

    # 整體回歸線
    slope, intercept, r, p, se = stats.linregress(age, y)
    xs = np.linspace(age.min()-0.5, age.max()+0.5, 200)
    ax.plot(xs, slope*xs + intercept, color=C_MAIN, linewidth=2,
            label=f"Overall trend  (r={r:.2f}, p={p:.3f})", zorder=4)

    # LOWESS 平滑線
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smooth = lowess(y, age, frac=0.4)
        ax.plot(smooth[:,0], smooth[:,1], color=C_ACCENT, linewidth=1.5,
                linestyle="--", label="LOWESS smooth", zorder=4)
    except ImportError:
        pass

    # 文獻參考線
    ax.axhline(0.13, color="#2E75B6", linestyle=":", linewidth=1.2, alpha=0.7,
               label="MiSight RCT median (0.13)")
    ax.axhline(0.36, color=C_GRAY, linestyle=":", linewidth=1.2, alpha=0.7,
               label="SV control (0.36)")

    # 圖例
    legend_patches = [
        mpatches.Patch(color=C_LIGHT,   label="MiSight only"),
        mpatches.Patch(color="#70AD47", label="+ Atropine 0.01%"),
        mpatches.Patch(color="#ED7D31", label="+ Atropine 0.05%"),
        mpatches.Patch(color=C_ACCENT,  label="+ Atropine 0.125%"),
    ]
    leg1 = ax.legend(handles=legend_patches, fontsize=8.5, loc="upper right",
                     title="Treatment", title_fontsize=8.5)
    ax.add_artist(leg1)
    ax.legend(fontsize=8.5, loc="upper left")

    ax.set_xlabel("Age at Treatment Start (years)", fontsize=11)
    ax.set_ylabel("Annualized AXL Progression (mm/yr)", fontsize=11)
    ax.set_title(
        "Age vs Myopia Progression Rate\n"
        "Colored by Adjunct Atropine Concentration",
        fontsize=12, fontweight="bold", pad=10
    )
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ {path.name}")


# ══════════════════════════════════════════════════════════
# 圖 3：Atropine 劑量效應
# ══════════════════════════════════════════════════════════
def plot_atropine_effect(df: pd.DataFrame, path: Path):
    df = df.copy()
    label_map = {0.0: "MiSight only\n(0%)", 0.01: "+Atropine\n0.01%",
                 0.02: "+Atropine\n0.02%", 0.05: "+Atropine\n0.05%",
                 0.125: "+Atropine\n0.125%"}
    order  = [0.0, 0.01, 0.02, 0.05, 0.125]
    colors = [C_LIGHT, "#A9D18E", "#70AD47", "#ED7D31", C_ACCENT]

    groups = []
    labels_plot = []
    colors_used = []
    counts = []
    medians = []

    for conc, col in zip(order, colors):
        grp = df[df["atropine_conc"] == conc]["y_annual_mm_yr"].dropna().values
        if len(grp) >= 3:
            groups.append(grp)
            labels_plot.append(label_map[conc])
            colors_used.append(col)
            counts.append(len(grp))
            medians.append(np.median(grp))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5),
                             gridspec_kw={"width_ratios":[1.8, 1]})

    # 左：Boxplot
    ax = axes[0]
    bp = ax.boxplot(groups, patch_artist=True, widths=0.55,
                    medianprops=dict(color="white", linewidth=2.5),
                    flierprops=dict(marker="o", markersize=4, alpha=0.5))
    for patch, c in zip(bp["boxes"], colors_used):
        patch.set_facecolor(c); patch.set_alpha(0.85)

    for i, (med, cnt) in enumerate(zip(medians, counts), start=1):
        ax.text(i, med + 0.025, f"{med:.3f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc=colors_used[i-1], alpha=0.85, ec="none"))
        ax.text(i, -0.62, f"n={cnt}", ha="center", fontsize=8, color="#555")

    ax.axhline(0.13, color="#2E75B6", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(len(groups)+0.4, 0.135, "MiSight\nRCT", fontsize=7.5, color="#2E75B6")
    ax.set_xticklabels(labels_plot, fontsize=9)
    ax.set_ylabel("Annualized AXL Progression (mm/yr)", fontsize=10)
    ax.set_title("Progression by Atropine Concentration", fontsize=11, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # 右：年齡中位數（確認各組年齡是否可比）
    ax2 = axes[1]
    age_meds = []
    age_iqr_lo, age_iqr_hi = [], []
    for conc in order:
        grp_age = df[df["atropine_conc"] == conc]["age_start"].dropna().values
        if len(grp_age) >= 3:
            age_meds.append(np.median(grp_age))
            age_iqr_lo.append(np.median(grp_age) - np.percentile(grp_age, 25))
            age_iqr_hi.append(np.percentile(grp_age, 75) - np.median(grp_age))

    xs = range(len(age_meds))
    ax2.bar(xs, age_meds, color=colors_used, edgecolor="white", width=0.6, alpha=0.85)
    ax2.errorbar(xs, age_meds,
                 yerr=[age_iqr_lo, age_iqr_hi],
                 fmt="none", color="#333", capsize=5, linewidth=1.5)
    for i, (m, lo, hi) in enumerate(zip(age_meds, age_iqr_lo, age_iqr_hi)):
        ax2.text(i, m + hi + 0.3, f"{m:.1f}", ha="center", fontsize=8.5,
                 fontweight="bold", color="#333")

    ax2.set_xticks(list(xs))
    ax2.set_xticklabels([l.replace("\n", " ") for l in labels_plot], fontsize=7.5, rotation=15)
    ax2.set_ylabel("Age at Start (years)", fontsize=10)
    ax2.set_title("Age Distribution\n(Confound Check)", fontsize=10, fontweight="bold")
    ax2.spines[["top","right"]].set_visible(False)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.set_ylim(0, max(age_meds) * 1.35)

    plt.suptitle("Atropine Dose–Response Analysis", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ {path.name}")


# ══════════════════════════════════════════════════════════
# 圖 4：升級版 RCT 對照（加年齡資訊）
# ══════════════════════════════════════════════════════════
def plot_rct_v2(df: pd.DataFrame, path: Path):
    """
    每個 bar 下方顯示：
      - 研究名稱
      - 年齡範圍 / 中位數
    讓讀者能判斷各研究是否可比。
    """
    y_real   = df["y_annual_mm_yr"].dropna().values
    age_real = df["age_start"].dropna().values

    real_med  = np.median(y_real)
    real_q25  = np.percentile(y_real, 25)
    real_q75  = np.percentile(y_real, 75)
    real_age_med = np.median(age_real)
    real_age_min = int(age_real.min())
    real_age_max = int(age_real.max())

    # (progression mm/yr, age_median, age_range_str, color, bar_label, ethnicity)
    studies = [
        (0.36,  10.0, "6–12",  C_GRAY,   "SV Control\n(Liu 2021)",           "East Asian\n(Singapore)"),
        (0.13,   9.9, "8–12", "#2E75B6", "MiSight 1yr\n(Chamberlain 2019)",  "Multi-ethnic\n(EU/US/NZ/SG)"),
        (0.15,  10.2, "8–12", "#1F4E79", "MiSight 3yr\n(Chamberlain 2019)",  "Multi-ethnic\n(EU/US/NZ/SG)"),
        (0.28,  10.0, "6–12", "#9DC3E6", "Atropine 0.01%\n(ATOM2 2012)",     "East Asian\n(Singapore)"),
        (0.19,   9.4, "6–12", "#70AD47", "Atropine 0.1%\n(LAMP 2019)",       "East Asian\n(Hong Kong)"),
        (real_med, real_age_med, f"{real_age_min}–{real_age_max}",
         C_ACCENT, f"This Study\n(Real-world)",                               "East Asian\n(Taiwan) ★"),
    ]

    labels     = [s[4] for s in studies]
    values     = [s[0] for s in studies]
    age_meds   = [s[1] for s in studies]
    age_rngs   = [s[2] for s in studies]
    colors     = [s[3] for s in studies]
    ethnicities= [s[5] for s in studies]

    fig, ax = plt.subplots(figsize=(11, 6))

    bars = ax.bar(range(len(studies)), values, color=colors,
                  edgecolor="white", width=0.6, zorder=3)

    # This Study 的 IQR 誤差棒
    this_i = len(studies) - 1
    ax.errorbar(this_i, real_med,
                yerr=[[real_med - real_q25], [real_q75 - real_med]],
                fmt="none", color="#7F0000", capsize=7, linewidth=2.2, zorder=4)

    # 進展數值標籤（bar 頂）
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.008,
                f"{val:.2f} mm/yr",
                ha="center", va="bottom", fontsize=9.5,
                fontweight="bold" if i == this_i else "normal",
                color=C_ACCENT if i == this_i else "#333")

    # X 軸標籤：研究名 + 年齡 + 族群
    xtick_labels = []
    for i, (lbl, age_m, age_r, eth) in enumerate(zip(labels, age_meds, age_rngs, ethnicities)):
        xtick_labels.append(f"{lbl}\nAge: {age_r} (med {age_m:.1f})\n{eth}")

    ax.set_xticks(range(len(studies)))
    ax.set_xticklabels(xtick_labels, fontsize=8.2)

    ax.set_ylabel("Annualized AXL Progression (mm/yr)", fontsize=11)
    ax.set_title(
        "Real-World Cohort vs Published RCT Benchmarks\n"
        "Axial Length Progression Under Myopia Management  |  Age & Ethnicity Context",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.set_ylim(0, max(values) * 1.35)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

    # 族群圖例
    ea_patch   = mpatches.Patch(facecolor="#FFF3CD", edgecolor="#999", label="East Asian cohort")
    multi_patch= mpatches.Patch(facecolor="#EAF4FB", edgecolor="#999", label="Multi-ethnic cohort")
    ax.legend(handles=[ea_patch, multi_patch], fontsize=8.5, loc="upper left")

    # East Asian bar 加底色標示
    for i, eth in enumerate(ethnicities):
        if "East Asian" in eth:
            bars[i].set_edgecolor("#B8860B")
            bars[i].set_linewidth(1.8)

    # 備注：This study IQR
    ax.text(this_i, real_q75 + 0.03,
            f"IQR [{real_q25:.2f}, {real_q75:.2f}]",
            ha="center", fontsize=7.5, color="#7F0000")

    # 資料來源 + 族群說明
    ax.text(0.01, 0.01,
            "Sources: Chamberlain 2019 (MiSight); Yam 2019 (LAMP); Chia 2012 (ATOM2); Liu 2021 (SV)\n"
            "Note: East Asian children show faster baseline progression than multi-ethnic Western cohorts — "
            "compare within same ethnicity for fair benchmarking.",
            transform=ax.transAxes, fontsize=7, color="#888",
            verticalalignment="bottom")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ {path.name}")


# ══════════════════════════════════════════════════════════
# 圖 5：SHAP Dependency — early_slope_6m 著色 by age
# ══════════════════════════════════════════════════════════
def plot_shap_dependency(df: pd.DataFrame, path: Path):
    if not HAS_SHAP:
        print("⚠️  shap 未安裝，跳過 SHAP dependency plot")
        return

    feats = [f for f in FEATURES_A if f in df.columns]
    sub   = df[feats + [TARGET]].dropna(subset=[TARGET])
    X     = sub[feats].values
    y     = sub[TARGET].values

    if HAS_XGB:
        from xgboost import XGBRegressor
        mdl = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("mdl", XGBRegressor(n_estimators=200, max_depth=3,
                                 learning_rate=0.05, random_state=42, verbosity=0)),
        ])
    else:
        mdl = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("mdl", RandomForestRegressor(n_estimators=200, random_state=42)),
        ])

    mdl.fit(X, y)
    imp = mdl.named_steps["imp"]
    X_imp = imp.transform(X)
    raw_mdl = mdl.named_steps["mdl"]

    try:
        booster = raw_mdl.get_booster()
        explainer = shap.TreeExplainer(booster)
        shap_vals = explainer.shap_values(X_imp)
    except Exception:
        bg = shap.sample(X_imp, 50, random_state=42)
        explainer = shap.KernelExplainer(raw_mdl.predict, bg)
        shap_vals = explainer.shap_values(X_imp, nsamples=100)

    feat_idx_slope = feats.index("early_slope_6m") if "early_slope_6m" in feats else None
    feat_idx_age   = feats.index("age_start")       if "age_start"       in feats else None

    if feat_idx_slope is None:
        print("⚠️  early_slope_6m 不在特徵集，跳過 SHAP dependency")
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))

    shap_slope = shap_vals[:, feat_idx_slope]
    x_slope    = X_imp[:, feat_idx_slope]
    age_vals   = X_imp[:, feat_idx_age] if feat_idx_age is not None else np.zeros(len(x_slope))

    sc = ax.scatter(x_slope, shap_slope,
                    c=age_vals, cmap="RdYlBu_r",
                    alpha=0.7, s=45, edgecolors="white", linewidths=0.3, zorder=3)
    cbar = fig.colorbar(sc, ax=ax, label="Age at start (years)")
    cbar.ax.tick_params(labelsize=9)

    ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="#888", linewidth=0.8, linestyle="--")

    ax.set_xlabel("Early AXL Slope 0–6m (mm/month)", fontsize=11)
    ax.set_ylabel("SHAP value\n(contribution to predicted mm/yr)", fontsize=11)
    ax.set_title(
        "SHAP Dependency: Early AXL Slope\n"
        "Colored by Age — Interaction Effect",
        fontsize=12, fontweight="bold", pad=10
    )
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(linestyle="--", alpha=0.3)

    # 說明文字
    ax.text(0.03, 0.95,
            "Higher slope → faster early progression\n"
            "Red = younger patients",
            transform=ax.transAxes, fontsize=8.5, color="#333",
            va="top", bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8, ec="#ccc"))

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ {path.name}")


# ══════════════════════════════════════════════════════════
# 圖 6：兩診所資料一致性
# ══════════════════════════════════════════════════════════
def plot_source_comparison(df: pd.DataFrame, path: Path):
    src_a = df[df["source"]=="大學"]["y_annual_mm_yr"].dropna().values
    src_b = df[df["source"]=="合安"]["y_annual_mm_yr"].dropna().values
    age_a = df[df["source"]=="大學"]["age_start"].dropna().values
    age_b = df[df["source"]=="合安"]["age_start"].dropna().values

    stat, pval = stats.mannwhitneyu(src_a, src_b, alternative="two-sided")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 左：進展分布
    ax = axes[0]
    ax.hist(src_a, bins=18, alpha=0.65, color=C_MAIN,   label=f"Clinic A (n={len(src_a)})", density=True)
    ax.hist(src_b, bins=14, alpha=0.65, color=C_ACCENT, label=f"Clinic B (n={len(src_b)})", density=True)
    ax.axvline(np.median(src_a), color=C_MAIN,   linestyle="--", linewidth=1.5,
               label=f"Median A: {np.median(src_a):.3f}")
    ax.axvline(np.median(src_b), color=C_ACCENT, linestyle="--", linewidth=1.5,
               label=f"Median B: {np.median(src_b):.3f}")
    ax.set_xlabel("Annualized AXL Progression (mm/yr)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"Progression Distribution\nMann-Whitney U  p = {pval:.3f}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(linestyle="--", alpha=0.3)

    # 右：年齡分布（確認兩診所年齡組成是否相似）
    ax2 = axes[1]
    ax2.hist(age_a, bins=14, alpha=0.65, color=C_MAIN,   label=f"Clinic A (n={len(age_a)})", density=True)
    ax2.hist(age_b, bins=10, alpha=0.65, color=C_ACCENT, label=f"Clinic B (n={len(age_b)})", density=True)
    ax2.axvline(np.median(age_a), color=C_MAIN,   linestyle="--", linewidth=1.5,
                label=f"Median A: {np.median(age_a):.1f} yrs")
    ax2.axvline(np.median(age_b), color=C_ACCENT, linestyle="--", linewidth=1.5,
                label=f"Median B: {np.median(age_b):.1f} yrs")
    ax2.set_xlabel("Age at Treatment Start (years)", fontsize=10)
    ax2.set_ylabel("Density", fontsize=10)
    ax2.set_title("Age Distribution\n(Clinic Comparability Check)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8.5)
    ax2.spines[["top","right"]].set_visible(False)
    ax2.grid(linestyle="--", alpha=0.3)

    plt.suptitle("Data Source Consistency — Two Clinics", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ {path.name}")


# ══════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════
def main():
    df = pd.read_csv(FEAT_FILE)
    print(f"讀入 features.csv：{len(df)} 隻眼\n")

    plot_age_boxplot(df,         FIG_DIR / "age_boxplot.png")
    plot_age_scatter(df,         FIG_DIR / "age_scatter.png")
    plot_atropine_effect(df,     FIG_DIR / "atropine_effect.png")
    plot_rct_v2(df,              FIG_DIR / "rct_benchmark_v2.png")
    plot_shap_dependency(df,     FIG_DIR / "shap_age_interaction.png")
    plot_source_comparison(df,   FIG_DIR / "source_comparison.png")

    print("\n✓ 全部 6 張圖完成，存在 figures/")


if __name__ == "__main__":
    main()
