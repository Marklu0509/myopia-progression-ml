"""
01_data_prep.py
───────────────
【這個檔案在做什麼】
讀入兩個診所的原始 Excel 檔，把它們清理成統一格式，
合併成一個乾淨的「long format」DataFrame，輸出 master.csv。

【為什麼這麼做】
- 兩個診所的欄位名稱/結構不同，必須個別解析再合併。
- 眼軸測量值散落在「月份編號」欄 (1, 2, 3, …)，
  需要 melt 成 long format 才能計算斜率/年化進展。
- 合安的姓名只在 OD 列有，OS 列是 NaN，需要 ffill()。
- 最後輸出每一「眼 × 時間點」一列的格式給 02_features.py 用。
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── 路徑設定 ──────────────────────────────────────────────
BASE = Path(__file__).parent.parent          # Data Base/
UNIV_FILE  = BASE / "datebase.xlsx"
HEAN_FILE  = BASE / "合安" / "合安病例紀錄.xlsx"
OUT_DIR    = Path(__file__).parent
OUT_MASTER = OUT_DIR / "master.csv"

# ══════════════════════════════════════════════════════════
# 1. 讀取「大學」診所
# ══════════════════════════════════════════════════════════
def load_university():
    """
    大學診所：
    - 月份欄是 Python int (1–16)，代表「第 N 個月」的眼軸測量
    - '開始時' 欄 = 療程 month 0（部分病人有，部分沒有）
    - 我們統一把 month 0 定為「開始時」若有，否則用月份欄最早的量測
    """
    df = pd.read_excel(UNIV_FILE, sheet_name="大學", header=0)

    # 只保留有意義的欄位
    id_cols  = ["姓名", "性別", "開始年齡", "西元", "左右眼", "近視", "散光", "Mi處方", "搭配其他療程"]
    month_cols = [c for c in df.columns if isinstance(c, int)]   # [1,2,...,16]
    axl0_col  = "開始時"   # month 0

    keep = id_cols + [axl0_col] + month_cols
    df = df[[c for c in keep if c in df.columns]].copy()

    # 把「開始時」加入 month 0
    df[0] = df[axl0_col]
    df.drop(columns=[axl0_col], inplace=True)

    # melt → long format
    all_month_cols = [0] + month_cols
    df_long = df.melt(
        id_vars=id_cols,
        value_vars=all_month_cols,
        var_name="month",
        value_name="axl_mm"
    )
    df_long["source"] = "大學"
    return df_long


# ══════════════════════════════════════════════════════════
# 2. 讀取「合安」診所
# ══════════════════════════════════════════════════════════
def load_hean():
    """
    合安診所：
    - sheet「全」，每位病人佔兩列 (OD/OS)，姓名/性別/年齡只在 OD 列
    - 月份欄同樣是 int (1–12)，代表「第 N 個月」
    - '起始(mm)' 欄 = month 0
    - 左右眼欄名是「左右眼.1」
    """
    df = pd.read_excel(HEAN_FILE, sheet_name="全", header=0)

    # ffill 姓名/性別/年齡（OS 列繼承 OD 列）
    for col in ["姓名", "性別", "年齡", "西元出生"]:
        if col in df.columns:
            df[col] = df[col].ffill()

    # 合安有兩個「左右眼」欄：原本那欄 + 「左右眼.1」
    # rename 前先 drop 原本那欄（兩欄內容相同），避免 rename 後重複欄名
    if "左右眼" in df.columns and "左右眼.1" in df.columns:
        df = df.drop(columns=["左右眼"])

    # datetime 欄轉字串，避免 pandas melt 在 ExtensionDtype 上的 bug
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)

    # 統一欄位名稱以利合併
    df = df.rename(columns={
        "左右眼.1"  : "左右眼",
        "年齡"      : "開始年齡",
        "西元出生"  : "西元",
        "起始(mm)"  : "axl_start",
    })

    # 近視度數：合安記在「度數」欄（取右眼值）
    if "度數" in df.columns:
        df["近視"] = df["度數"]
    if "Mi處方" not in df.columns:
        df["Mi處方"] = np.nan

    id_cols    = ["姓名", "性別", "開始年齡", "西元", "左右眼", "近視", "散光", "Mi處方", "搭配其他療程"]
    id_cols    = [c for c in id_cols if c in df.columns]
    month_cols = [c for c in df.columns if isinstance(c, (int, np.integer)) and c not in id_cols]

    df[0] = df["axl_start"] if "axl_start" in df.columns else np.nan

    all_month_cols = [0] + month_cols
    df_long = df.melt(
        id_vars=id_cols,
        value_vars=all_month_cols,
        var_name="month",
        value_name="axl_mm"
    )
    df_long["source"] = "合安"
    return df_long


# ══════════════════════════════════════════════════════════
# 3. 合併 + 清理
# ══════════════════════════════════════════════════════════
def clean_axl(val):
    """把非數字值（'-', '無', NaN, 0）過濾掉，0 視為遺漏"""
    try:
        v = float(val)
        return v if (v > 15) else np.nan   # 眼軸合理範圍 > 15 mm
    except (ValueError, TypeError):
        return np.nan


def clean_atropine(val):
    """
    把搭配其他療程欄統一成 atropine 濃度數值：
    0.01A / 0.01 → 0.01
    0.05         → 0.05
    0.125        → 0.125
    停 / 停OK / NaN → 0.0
    """
    if pd.isna(val):
        return 0.0
    s = str(val).strip().lower()
    if "0.125" in s:
        return 0.125
    if "0.05" in s:
        return 0.05
    if "0.01" in s:
        return 0.01
    if "0.02" in s:
        return 0.02
    return 0.0


def main():
    print("讀取大學資料…")
    df_u = load_university()
    print(f"  大學 long rows: {len(df_u)}")

    print("讀取合安資料…")
    df_h = load_hean()
    print(f"  合安 long rows: {len(df_h)}")

    # 合併
    df = pd.concat([df_u, df_h], ignore_index=True)

    # 清理眼軸值
    df["axl_mm"] = df["axl_mm"].apply(clean_axl)

    # 只保留有眼軸量測的列
    df = df[df["axl_mm"].notna()].copy()

    # 清理 atropine 濃度
    df["atropine_conc"] = df["搭配其他療程"].apply(clean_atropine)

    # Mi處方：清理成數值（度數），NaN → 0
    df["mi_rx"] = pd.to_numeric(df["Mi處方"], errors="coerce").fillna(0)

    # 開始年齡：確保是數值
    df["開始年齡"] = pd.to_numeric(df["開始年齡"], errors="coerce")

    # 近視度數：確保是數值
    df["近視"] = pd.to_numeric(df["近視"], errors="coerce")

    # month 確保是 int
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")

    # 建立唯一眼識別碼
    df["eye_id"] = (
        df["source"].astype(str) + "_" +
        df["姓名"].astype(str) + "_" +
        df["左右眼"].astype(str)
    )

    # 性別 → 0/1
    df["sex"] = df["性別"].map({"M": 1, "F": 0})

    # 左右眼 → 0/1
    df["laterality"] = df["左右眼"].map({"OD": 0, "OS": 1})

    # 最終欄位順序
    final_cols = [
        "eye_id", "source", "姓名", "性別", "sex",
        "開始年齡", "laterality", "左右眼",
        "近視", "mi_rx", "atropine_conc",
        "month", "axl_mm"
    ]
    final_cols = [c for c in final_cols if c in df.columns]
    df = df[final_cols].sort_values(["eye_id", "month"]).reset_index(drop=True)

    # 統計
    n_eyes = df["eye_id"].nunique()
    print(f"\n合併後：{n_eyes} 隻眼，{len(df)} 筆測量")

    eyes_ge2 = df.groupby("eye_id").size()
    print(f"  ≥2 個時間點：{(eyes_ge2 >= 2).sum()} 隻眼")
    print(f"  ≥3 個時間點：{(eyes_ge2 >= 3).sum()} 隻眼")
    print(f"  ≥4 個時間點：{(eyes_ge2 >= 4).sum()} 隻眼")

    df.to_csv(OUT_MASTER, index=False, encoding="utf-8-sig")
    print(f"\n✓ 輸出：{OUT_MASTER}")


if __name__ == "__main__":
    main()
