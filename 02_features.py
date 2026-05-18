"""
02_features.py
──────────────
【這個檔案在做什麼】
吃進 master.csv（每眼 × 時間點 的 long format），
對每一隻眼計算一組臨床+統計特徵，以及「年化進展 mm/yr」target y，
輸出 features.csv（每眼一列）。

【為什麼這麼做】
ML 模型需要「每個樣本一列」的格式（wide format）。
特徵設計分三類：
  1. 「短期斜率/變異度」→ 捕捉治療早期反應，是預測長期的核心
  2. 「基線臨床值」→ 年齡、近視度數等背景因子
  3. 「治療處方」→ MiSight 度數、atropine 濃度

Target y = 年化眼軸進展 (mm/yr)：
  用 OLS 線性回歸對「所有時間點」擬合斜率，再 × 12 換算為年化值。
  這樣可以利用所有 ≥2 點的眼，最大化樣本量。

短期斜率 (early_slope_3m / early_slope_6m)：
  只用前 3 或前 6 個月的量測算 OLS 斜率 (mm/月)，
  代表治療初期眼軸增速，是最強的預測特徵之一。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

IN_FILE  = Path(__file__).parent / "master.csv"
OUT_FILE = Path(__file__).parent / "features.csv"


# ── 年齡-眼軸常模 (文獻值，用於計算偏離量) ─────────────────────
# 來源：He X et al. Br J Ophthalmol 2023;107:167–175
#       中國兒童 4–18 歲軸長百分位曲線（上海+廣州，n=14,127）
#       使用 50th percentile（中位數）分性別，以便與東亞兒童做準確比較
#       sex=1 (Male), sex=0 (Female)
#
# 設計說明：因台灣資料有 sex 欄位，這裡改用分性別常模，
# 若 sex 未知則取男女平均作為 fallback。
AGE_AXL_NORM_MALE = {
    4: 22.39, 5: 22.69, 6: 22.97, 7: 23.25, 8: 23.51,
    9: 23.76, 10: 23.99, 11: 24.22, 12: 24.43, 13: 24.62,
    14: 24.81, 15: 24.98, 16: 25.13, 17: 25.28, 18: 25.41,
    19: 25.50, 20: 25.55,  # 18歲後趨於穩定，以18歲值外插
}
AGE_AXL_NORM_FEMALE = {
    4: 21.78, 5: 22.10, 6: 22.41, 7: 22.70, 8: 22.98,
    9: 23.25, 10: 23.51, 11: 23.75, 12: 23.97, 13: 24.19,
    14: 24.39, 15: 24.57, 16: 24.75, 17: 24.91, 18: 25.05,
    19: 25.12, 20: 25.15,  # 18歲後趨於穩定，以18歲值外插
}

def get_norm_axl(age_int: int, sex: int) -> float:
    """取東亞兒童眼軸常模（He et al. 2023）。sex=1男, sex=0女, 其他取平均。"""
    if sex == 1:
        return AGE_AXL_NORM_MALE.get(age_int, AGE_AXL_NORM_MALE.get(18, 25.41))
    elif sex == 0:
        return AGE_AXL_NORM_FEMALE.get(age_int, AGE_AXL_NORM_FEMALE.get(18, 25.05))
    else:  # 未知性別 → 男女平均
        m = AGE_AXL_NORM_MALE.get(age_int, 25.41)
        f = AGE_AXL_NORM_FEMALE.get(age_int, 25.05)
        return (m + f) / 2


def ols_slope(months, axls):
    """最小平方回歸斜率 (mm/月)，需 ≥2 個有效點。"""
    m = np.array(months, dtype=float)
    a = np.array(axls, dtype=float)
    mask = np.isfinite(m) & np.isfinite(a)
    if mask.sum() < 2:
        return np.nan
    slope, *_ = np.polyfit(m[mask], a[mask], 1)
    return slope


def compute_features(grp: pd.DataFrame) -> dict:
    """給單眼的 long-format 資料，回傳特徵 dict。"""
    grp = grp.sort_values("month").copy()
    months = grp["month"].values.astype(float)
    axls   = grp["axl_mm"].values.astype(float)

    valid_mask = np.isfinite(months) & np.isfinite(axls)
    m_v = months[valid_mask]
    a_v = axls[valid_mask]
    n   = len(m_v)

    if n < 2:
        return None   # 不足 2 點，丟棄

    # ── Target y：年化進展 (mm/yr) ────────────────────────────
    full_slope_per_month = ols_slope(m_v, a_v)          # mm/月
    y_annual             = full_slope_per_month * 12     # mm/yr

    # ── 基線眼軸 (month 0 或最早量測) ────────────────────────
    axl_baseline = a_v[0]

    # ── 短期斜率 ──────────────────────────────────────────────
    # 前 3 個月 (month ≤ 3)
    mask_3m = m_v <= 3
    early_slope_3m = ols_slope(m_v[mask_3m], a_v[mask_3m]) if mask_3m.sum() >= 2 else np.nan

    # 前 6 個月 (month ≤ 6)
    mask_6m = m_v <= 6
    early_slope_6m = ols_slope(m_v[mask_6m], a_v[mask_6m]) if mask_6m.sum() >= 2 else np.nan

    # ── 短期測量變異度 ────────────────────────────────────────
    axl_std_6m = a_v[mask_6m].std() if mask_6m.sum() >= 2 else np.nan

    # ── 年齡偏離量 ────────────────────────────────────────────
    # 以開始年齡+性別查東亞常模（He et al. 2023），計算 baseline AXL 偏離
    age = grp["開始年齡"].iloc[0]
    age_int = int(round(age)) if pd.notna(age) else None
    sex_val  = int(grp.iloc[0]["sex"]) if pd.notna(grp.iloc[0].get("sex")) else -1
    norm_axl = get_norm_axl(age_int, sex_val) if age_int else np.nan
    axl_dev  = axl_baseline - norm_axl              # + 表示超過常模（近視較深）

    # ── 臨床基線特徵 ──────────────────────────────────────────
    row0 = grp.iloc[0]
    myopia_d     = float(row0["近視"])    if pd.notna(row0.get("近視"))    else np.nan
    mi_rx        = float(row0["mi_rx"])   if pd.notna(row0.get("mi_rx"))   else 0.0
    atropine     = float(row0["atropine_conc"]) if pd.notna(row0.get("atropine_conc")) else 0.0
    sex          = int(row0["sex"])       if pd.notna(row0.get("sex"))       else np.nan
    laterality   = int(row0["laterality"]) if pd.notna(row0.get("laterality")) else np.nan

    return {
        # 識別碼
        "eye_id"         : row0["eye_id"],
        "source"         : row0["source"],
        "左右眼"         : row0.get("左右眼", np.nan),
        # 基線臨床
        "age_start"      : float(age) if pd.notna(age) else np.nan,
        "sex"            : sex,
        "laterality"     : laterality,
        "myopia_d"       : myopia_d,
        "axl_baseline"   : axl_baseline,
        "axl_dev_from_norm": axl_dev,
        # 治療處方
        "mi_rx"          : mi_rx,
        "atropine_conc"  : atropine,
        # 短期動態特徵
        "early_slope_3m" : early_slope_3m,   # mm/月
        "early_slope_6m" : early_slope_6m,   # mm/月
        "axl_std_6m"     : axl_std_6m,
        # 資料品質
        "n_measurements" : n,
        # Target
        "y_annual_mm_yr" : y_annual,
    }


def main():
    df = pd.read_csv(IN_FILE)
    print(f"讀入 master.csv：{df['eye_id'].nunique()} 隻眼，{len(df)} 筆")

    records = []
    for eye_id, grp in df.groupby("eye_id"):
        feat = compute_features(grp)
        if feat is not None:
            records.append(feat)

    feat_df = pd.DataFrame(records)

    # 過濾 y 極端值 (|y| > 1.5 mm/yr 幾乎不可能是真實值，視為資料錯誤)
    before = len(feat_df)
    feat_df = feat_df[feat_df["y_annual_mm_yr"].abs() <= 1.5].copy()
    print(f"過濾極端 y 後：{before} → {len(feat_df)} 隻眼")

    # 統計
    print(f"\n特徵矩陣：{feat_df.shape}")
    print(f"  y (mm/yr) 中位數: {feat_df['y_annual_mm_yr'].median():.3f}")
    print(f"  y (mm/yr) 平均:   {feat_df['y_annual_mm_yr'].mean():.3f}")
    print(f"  y 範圍: [{feat_df['y_annual_mm_yr'].min():.3f}, {feat_df['y_annual_mm_yr'].max():.3f}]")

    # 缺失值報告
    print("\n缺失值比例：")
    for col in feat_df.columns:
        miss = feat_df[col].isna().mean()
        if miss > 0:
            print(f"  {col}: {miss:.1%}")

    feat_df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n✓ 輸出：{OUT_FILE}")


if __name__ == "__main__":
    main()
